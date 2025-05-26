from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Añadir esta importación
from typing import Optional, List
import ollama
from pydantic import BaseModel, Field
import yaml
from llama_index.llms.ollama import Ollama
from rag import RAG
from model_comparison import compare_models, CompareRequest


config_file = "config.yml"

with open(config_file, "r") as conf:
    config = yaml.safe_load(conf)


class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)


class Response(BaseModel):
    search_result: str 
    source: str


# Nueva clase para la solicitud de chat
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None  # Añadir campo opcional para el modelo
    collection: Optional[str] = None  # Añadir campo opcional para la colección


llm = Ollama(model=config["llm_name"], url=config["llm_url"], request_timeout=300.0)
rag = RAG(config_file=config, llm=llm)
index = rag.qdrant_index()


app = FastAPI()

# Añadir middleware CORS para que el frontend pueda conectarse
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Research RAG"}

a = "You can only answer based on the provided context. If a response cannot be formed strictly using the context, politely say you don't have knowledge about that topic"

@app.post("/api/search", response_model=Response, status_code=200)
def search(query: Query):
    query_engine = index.as_query_engine(similarity_top_k=query.similarity_top_k, output=Response, response_mode="tree_summarize", verbose=True)
    response = query_engine.query(query.query + a)
    response_object = Response(
        search_result=str(response).strip(), source=[response.metadata[k]["file_path"] for k in response.metadata.keys()][0]
    )
    return response_object

# Nuevo endpoint para el chat que usará tu sistema RAG existente
@app.post("/chat")
async def process_chat(request: ChatRequest):
    try:
        # Usar el modelo especificado o el por defecto
        selected_model = request.model or config["llm_name"]
        
        # Usar la colección especificada o la por defecto
        selected_collection = request.collection or config["collection_name"]
        
        # Crear LLM con el modelo seleccionado
        llm_for_query = Ollama(model=selected_model, url=config["llm_url"], request_timeout=300.0)
        
        # Crear configuración temporal con la colección seleccionada
        temp_config = config.copy()
        temp_config["collection_name"] = selected_collection
        
        # Crear RAG temporal con el modelo y colección seleccionados
        temp_rag = RAG(config_file=temp_config, llm=llm_for_query)
        temp_index = temp_rag.qdrant_index()
        
        # Crear una consulta usando el mensaje del chat
        query = Query(query=request.message)
        
        # Utilizar el motor de búsqueda con el modelo y colección seleccionados
        query_engine = temp_index.as_query_engine(similarity_top_k=1, response_mode="tree_summarize", verbose=True)
        response = query_engine.query(request.message + a)
        
        # Formatear respuesta para el frontend
        ai_response = str(response).strip()
        
        return {
            "response": ai_response, 
            "model_used": selected_model,
            "collection_used": selected_collection
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el mensaje: {str(e)}")

# Añadir este nuevo endpoint (sin modificar ninguno existente)
@app.post("/compare-models")
async def api_compare_models(request: CompareRequest):
    """Compara respuestas de múltiples modelos para una misma consulta."""
    try:
        result = compare_models(request, config_file, a)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al comparar modelos: {str(e)}")

# Endpoint para obtener modelos disponibles
@app.get("/api/models")
def get_models():
    try:
        from model_comparison import get_available_models
        models = get_available_models(config)
        default_model = config["llm_name"]
        return {
            "models": models,
            "default_model": default_model
        }
    except Exception as e:
        # Fallback en caso de error
        return {
            "models": [config["llm_name"]],
            "default_model": config["llm_name"]
        }

# Nuevos endpoints para gestión de colecciones
@app.get("/api/collections")
def get_collections():
    """Obtiene la lista de colecciones disponibles en Qdrant."""
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url=config["qdrant_url"])
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        return {
            "collections": collection_names,
            "current": config["collection_name"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener colecciones: {str(e)}")

@app.post("/api/collections")
def create_collection(request: dict):
    """Crea una nueva colección en Qdrant."""
    try:
        name = request.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Nombre de colección requerido")
            
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance
        
        client = QdrantClient(url=config["qdrant_url"])
        
        # Obtener dimensiones de embeddings según el modelo
        embedding_model = config["embedding_model"]
        # Puedes tener un diccionario de dimensiones por modelo o usar una API
        dimensions = 1024  # Ajustar según el modelo real
        
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE)
        )
        
        return {"message": f"Colección '{name}' creada correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al crear colección: {str(e)}")

@app.post("/api/documents/process")
async def process_documents(request: dict):
    """Procesa documentos y los carga en una colección específica."""
    try:
        collection = request.get("collection")
        directory = request.get("directory")
        chunk_size = request.get("chunk_size", config["chunk_size"])
        
        if not collection or not directory:
            raise HTTPException(status_code=400, detail="Colección y directorio requeridos")
            
        # Llamada asíncrona al procesador de documentos
        import subprocess
        import os
        
        process = subprocess.Popen([
            "python", "-m", "src.process_documents",
            "--directory", directory,
            "--collection", collection,
            "--chunk_size", str(chunk_size)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Puedes devolver inmediatamente un ID de trabajo para consultar el estado después
        return {"job_id": str(process.pid), "message": "Procesamiento iniciado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar documentos: {str(e)}")

@app.get("/api/documents/status/{job_id}")
def check_processing_status(job_id: str):
    """Verifica el estado de un trabajo de procesamiento."""
    # Implementar lógica para verificar el estado
    # ...
