from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import ollama
from pydantic import BaseModel, Field
import yaml
from llama_index.llms.ollama import Ollama
from rag import RAG
from model_comparison import compare_models, CompareRequest
import logging

config_file = "config.yml"

with open(config_file, "r") as conf:
    config = yaml.safe_load(conf)

class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)

class Response(BaseModel):
    search_result: str 
    source: str

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    collection: Optional[str] = None

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

llm = Ollama(model=config["llm_name"], url=config["llm_url"], request_timeout=300.0)
rag = RAG(config_file=config, llm=llm)
index = rag.qdrant_index()

# Cache para las instancias RAG por modelo y colección
rag_cache = {}

def get_or_create_rag(model_name: str, collection_name: str):
    """Obtiene una instancia RAG del cache o crea una nueva si no existe."""
    cache_key = f"{model_name}_{collection_name}"
    
    if cache_key not in rag_cache:
        logger.info(f"Creando nueva instancia RAG para {model_name} - {collection_name}")
        
        # Crear LLM
        llm_instance = Ollama(model=model_name, url=config["llm_url"], request_timeout=300.0)
        
        # Crear configuración temporal
        temp_config = config.copy()
        temp_config["collection_name"] = collection_name
        
        # Crear RAG
        rag_instance = RAG(config_file=temp_config, llm=llm_instance)
        index_instance = rag_instance.qdrant_index()
        
        rag_cache[cache_key] = {
            'rag': rag_instance,
            'index': index_instance,
            'llm': llm_instance
        }
        
        logger.info(f"Instancia RAG creada y cacheada para {cache_key}")
    else:
        logger.info(f"Reutilizando instancia RAG cacheada para {cache_key}")
    
    return rag_cache[cache_key]

app = FastAPI()

# Añadir middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
        search_result=str(response).strip(), 
        source=[response.metadata[k]["file_path"] for k in response.metadata.keys()][0]
    )
    return response_object

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error no manejado en {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Error interno del servidor: {str(exc)}",
            "type": "internal_error",
            "path": str(request.url)
        }
    )

@app.post("/chat")
async def process_chat(request: ChatRequest):
    try:
        logger.info(f"=== INICIO CONSULTA ===")
        logger.info(f"Mensaje: {request.message[:100]}...")
        
        # Usar el modelo especificado o el por defecto
        selected_model = request.model or config["llm_name"]
        selected_collection = request.collection or config["collection_name"]
        
        logger.info(f"Modelo: {selected_model}, Colección: {selected_collection}")
        
        # Obtener o crear instancia RAG cacheada
        rag_instances = get_or_create_rag(selected_model, selected_collection)
        
        logger.info("Ejecutando consulta...")
        query_engine = rag_instances['index'].as_query_engine(
            similarity_top_k=1, 
            response_mode="tree_summarize", 
            verbose=True
        )
        
        response = query_engine.query(request.message + a)
        logger.info("Consulta ejecutada correctamente")
        
        # Formatear respuesta
        ai_response = str(response).strip()
        logger.info(f"Respuesta generada (longitud: {len(ai_response)})")
        
        result = {
            "response": ai_response, 
            "model_used": selected_model,
            "collection_used": selected_collection
        }
        
        logger.info("=== FIN CONSULTA EXITOSA ===")
        return result
        
    except Exception as e:
        logger.error(f"=== ERROR EN CONSULTA ===")
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar el mensaje: {str(e)}")

@app.post("/compare-models")
async def api_compare_models(request: CompareRequest):
    """Compara respuestas de múltiples modelos para una misma consulta."""
    try:
        result = compare_models(request, config_file, a)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al comparar modelos: {str(e)}")

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
        return {
            "models": [config["llm_name"]],
            "default_model": config["llm_name"]
        }

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
            
        import subprocess
        
        process = subprocess.Popen([
            "python", "-m", "src.process_documents",
            "--directory", directory,
            "--collection", collection,
            "--chunk_size", str(chunk_size)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return {"job_id": str(process.pid), "message": "Procesamiento iniciado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar documentos: {str(e)}")

@app.get("/api/documents/status/{job_id}")
def check_processing_status(job_id: str):
    """Verifica el estado de un trabajo de procesamiento."""
    # Implementar lógica para verificar el estado
    return {"status": "pending", "job_id": job_id}

@app.post("/api/cache/clear")
def clear_cache():
    """Limpia el cache de instancias RAG."""
    global rag_cache
    rag_cache.clear()
    return {"message": "Cache limpiado correctamente"}

@app.get("/api/status")
def get_status():
    """Obtiene el estado del servidor y cache."""
    return {
        "rag_cache_keys": list(rag_cache.keys()),
        "cache_size": len(rag_cache),
        "memory_usage": "Disponible en futuras versiones"
    }
