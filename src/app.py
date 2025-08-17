# ===== CONFIGURACIÓN AUTOMÁTICA DE CACHE =====
from cache_manager import initialize_cache, get_cache_manager

# Configurar cache antes de importar FastEmbed
cache_info = initialize_cache()
cache_manager = get_cache_manager()

# ===== RESTO DE IMPORTS =====
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import ollama
from pydantic import BaseModel, Field
import yaml
from llama_index.llms.ollama import Ollama
from rag import RAG
from model_comparison import CompareRequest  # ← Remover compare_models, solo CompareRequest
from llama_index.core.settings import Settings
import logging
import tempfile
import os
import shutil


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
    chunk_size: Optional[int] = None  # Nuevo campo opcional

class CompareModelsRequest(BaseModel):
    message: str
    models: List[str]
    collection: str
    judge_model: str
    include_retrieval_metrics: bool = False
    include_ragas_metrics: bool = False  # NUEVO CAMPO

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache para las instancias RAG por modelo y colección
rag_cache = {}

def get_or_create_rag(model_name: str, collection_name: str, chunk_size: int = 1024):
    """Obtiene una instancia RAG del cache o crea una nueva si no existe."""
    cache_key = f"{model_name}_{collection_name}_{chunk_size}"  # Incluir chunk_size en cache
    
    if cache_key not in rag_cache:
        logger.info(f"Creando nueva instancia RAG para {model_name} - {collection_name} - chunk_size: {chunk_size}")
        
        # Crear configuración temporal
        temp_config = config.copy()
        temp_config["collection_name"] = collection_name
        temp_config["chunk_size"] = chunk_size  # Configurar dinámicamente
        
        # Asegurar que el modelo de embeddings esté listo y configurado
        cache_manager.ensure_embedding_model_ready(temp_config)
        
        # Crear LLM
        llm_instance = Ollama(model=model_name, url=config["llm_url"], request_timeout=300.0)
        
        # Crear RAG sin embed_model - usará automáticamente el cache configurado
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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",  # Vite por defecto
        "http://localhost:4173",  # Vite preview
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173"
    ],
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
        
        # Validar que se proporcionen modelo y colección
        if not request.model:
            raise HTTPException(status_code=400, detail="El modelo LLM es obligatorio")
        if not request.collection:
            raise HTTPException(status_code=400, detail="La colección es obligatoria")
        
        selected_model = request.model
        selected_collection = request.collection
        selected_chunk_size = request.chunk_size or 1024  # Valor por defecto
        
        logger.info(f"Modelo: {selected_model}, Colección: {selected_collection}")
        
        # Obtener o crear instancia RAG cacheada
        rag_instances = get_or_create_rag(selected_model, selected_collection, selected_chunk_size)
        
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
        
    except HTTPException:
        raise  # Re-lanzar HTTPException tal como está
    except Exception as e:
        logger.error(f"=== ERROR EN CONSULTA ===")
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar el mensaje: {str(e)}")

@app.post("/compare-models")
async def compare_models(request: CompareRequest):
    """Compara respuestas usando evaluación académica LlamaIndex con modelo juez."""
    try:
        logger.info("=== INICIANDO EVALUACIÓN ACADÉMICA ===")
        
        # Validaciones
        if not request.models or len(request.models) < 1:
            raise HTTPException(status_code=400, detail="Se requiere al menos un modelo para la comparación")
        
        if not request.collection:
            raise HTTPException(status_code=400, detail="La colección es obligatoria")
        
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")
        
        if not request.judge_model:
            raise HTTPException(status_code=400, detail="El modelo juez es obligatorio")
        
        # Validar que juez no esté en modelos a comparar
        if request.judge_model in request.models:
            raise HTTPException(status_code=400, detail="El modelo juez no puede estar en la lista de modelos a comparar")
        
        logger.info(f"Modelos a evaluar: {request.models}")
        logger.info(f"Modelo juez: {request.judge_model}")
        logger.info(f"Colección: {request.collection}")
        logger.info(f"Incluir métricas de retrieval: {request.include_retrieval_metrics}")
        
        # Usar evaluación académica con LlamaIndex
        from model_comparison import academic_llamaindex_evaluation
        
        config = {
            "similarity_threshold": 0.7,
            "max_retrievals": 5,
            "include_ragas_metrics": request.include_ragas_metrics,  # NUEVO PARÁMETRO
            "embedding_model": config["embedding_model"]
        }
        
        result = academic_llamaindex_evaluation(request, config)
        
        if "error" in result:
            logger.error(f"Error en evaluación: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
        
        logger.info("=== EVALUACIÓN ACADÉMICA COMPLETADA ===")
        
        # ✅ DEVOLVER DIRECTAMENTE EL RESULTADO - más simple y sin errores
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en evaluación académica: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al comparar modelos: {str(e)}")

@app.get("/api/models")
def get_models():
    try:
        from model_comparison import get_available_models
        models = get_available_models(config)
        return {
            "models": models,
            "default_model": None  # Sin valor por defecto
        }
    except Exception as e:
        # Si falla, devolver lista vacía
        return {
            "models": [],
            "default_model": None
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
            "current": None  # Sin valor por defecto
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

@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection: str = Form(...),
    chunk_size: int = Form(default=1024)
):
    """Sube archivos y los procesa usando docling + Qdrant."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No se enviaron archivos")
        
        # Crear directorio temporal
        temp_dir = tempfile.mkdtemp()
        uploaded_filenames = []
        
        try:
            # Guardar archivos en directorio temporal
            for file in files:
                if file.filename:
                    file_path = os.path.join(temp_dir, file.filename)
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    uploaded_filenames.append(file.filename)
                    print(f"📁 Archivo guardado: {file.filename}")
            
            print(f"🔄 Iniciando procesamiento de {len(uploaded_filenames)} archivos...")
            
            # Ejecutar el script process_documents.py directamente con python
            import subprocess
            import sys
            
            # Obtener el path del script process_documents.py
            script_path = os.path.join(os.path.dirname(__file__), "process_documents.py")
            
            result = subprocess.run([
                sys.executable,  # Usar el mismo Python que está ejecutando la app
                script_path,
                "--directory", temp_dir,
                "--collection", collection,
                "--chunk_size", str(chunk_size)
            ], capture_output=True, text=True, timeout=1800)  # ← 30 minutos (era 300)
            
            if result.returncode == 0:
                print("✅ Procesamiento completado exitosamente")
                print(f"Output: {result.stdout}")
                return {
                    "message": f"Archivos procesados correctamente: {', '.join(uploaded_filenames)}",
                    "files_processed": uploaded_filenames,
                    "collection": collection,
                    "status": "completed"
                }
            else:
                print(f"❌ Error en procesamiento: {result.stderr}")
                print(f"Return code: {result.returncode}")
                print(f"Stdout: {result.stdout}")
                raise Exception(f"Error en procesamiento: {result.stderr}")
            
        finally:
            # Limpiar directorio temporal
            print(f"🧹 Limpiando directorio temporal: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="El procesamiento tardó demasiado tiempo")
    except Exception as e:
        print(f"💥 Error en upload_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar archivos: {str(e)}")
