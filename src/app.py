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
        # Crear una consulta usando el mensaje del chat
        query = Query(query=request.message)
        
        # Utilizar tu motor de búsqueda existente
        query_engine = index.as_query_engine(similarity_top_k=1, response_mode="tree_summarize", verbose=True)
        response = query_engine.query(request.message + a)
        
        # Formatear respuesta para el frontend
        ai_response = str(response).strip()
        
        return {"response": ai_response}
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
