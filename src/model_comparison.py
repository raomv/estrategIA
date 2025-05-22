from fastapi import HTTPException
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from typing import Dict, List, Optional
import yaml
import ollama

# Importar tu clase RAG
from rag import RAG

# Importaciones para RAGAS
try:
    from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    print("RAGAS no está instalado. Las métricas de evaluación no estarán disponibles.")
    RAGAS_AVAILABLE = False

class CompareRequest(BaseModel):
    message: str
    models: Optional[List[str]] = None  # Opcional: lista de modelos a comparar

def get_available_models(config):
    """Obtiene la lista de modelos disponibles en Ollama usando la API REST."""
    try:
        import requests
        
        # Construir la URL correcta para la API de Ollama
        ollama_url = config["llm_url"].rstrip("/")
        tags_url = f"{ollama_url}/api/tags"
        
        print(f"Consultando modelos disponibles en: {tags_url}")
        
        # Realizar la petición HTTP
        response = requests.get(tags_url)
        
        if response.status_code == 200:
            data = response.json()
            if "models" in data and isinstance(data["models"], list):
                # Extraer los nombres de los modelos
                models = [model["name"] for model in data["models"]]
                print(f"Modelos encontrados: {models}")
                return models
        
        # Si llegamos aquí, algo falló - usar el modelo por defecto
        print(f"No se pudieron obtener modelos de {tags_url}")
        return [config["llm_name"]]
        
    except Exception as e:
        import traceback
        print(f"Error al obtener modelos: {str(e)}")
        print(traceback.format_exc())
        return [config["llm_name"]]  # Fallback al modelo por defecto

def compare_models(request: CompareRequest, config_file: str, prompt_suffix: str):
    """
    Compara respuestas de múltiples modelos para la misma consulta.
    """
    try:
        # Cargar configuración
        with open(config_file, "r") as conf:
            config = yaml.safe_load(conf)
        
        # Determinar qué modelos comparar
        if request.models and len(request.models) > 0:
            models_to_compare = request.models
        else:
            # Obtener todos los modelos disponibles en Ollama
            models_to_compare = get_available_models(config)
        
        results = {}
        metrics = {}
        contexts = {}
        
        # Obtener respuesta de cada modelo
        for model_name in models_to_compare:
            try:
                # Crear instancia de Ollama con el modelo específico
                current_llm = Ollama(model=model_name, url=config["llm_url"], request_timeout=300.0)
                temp_rag = RAG(config_file=config, llm=current_llm)
                temp_index = temp_rag.qdrant_index()
                
                # Generar respuesta con este modelo
                query_engine = temp_index.as_query_engine(similarity_top_k=1, response_mode="tree_summarize", verbose=True)
                response = query_engine.query(request.message + prompt_suffix)
                
                # Guardar la respuesta
                ai_response = str(response).strip()
                results[model_name] = ai_response
                
                # Extraer el contexto recuperado para evaluación
                retrieved_context = [node.text for node in response.source_nodes]
                contexts[model_name] = retrieved_context
                
                # Evaluar con RAGAS si está disponible
                if RAGAS_AVAILABLE:
                    eval_result = evaluate(
                        questions=[request.message],
                        contexts=[retrieved_context],
                        answers=[ai_response],
                        metrics=[context_relevancy, answer_relevancy, faithfulness]
                    )
                    
                    # Guardar métricas para este modelo
                    metrics[model_name] = {
                        "context_relevancy": float(eval_result["context_relevancy"].iloc[0]),
                        "answer_relevancy": float(eval_result["answer_relevancy"].iloc[0]),
                        "faithfulness": float(eval_result["faithfulness"].iloc[0]),
                        # Promedio de las métricas como "puntuación general"
                        "overall_score": float((eval_result["context_relevancy"].iloc[0] + 
                                         eval_result["answer_relevancy"].iloc[0] + 
                                         eval_result["faithfulness"].iloc[0]) / 3)
                    }
                else:
                    # Si RAGAS no está disponible, proporcionar métricas vacías
                    metrics[model_name] = {
                        "context_relevancy": None,
                        "answer_relevancy": None,
                        "faithfulness": None,
                        "overall_score": None,
                        "note": "RAGAS no está instalado. Instala RAGAS para ver métricas de evaluación."
                    }
                    
            except Exception as e:
                # Si hay un error con un modelo, continuar con los demás
                results[model_name] = f"Error al usar este modelo: {str(e)}"
                metrics[model_name] = {"error": str(e)}
        
        return {
            "results": results,
            "metrics": metrics,
            "contexts": contexts  # Opcional: incluir contextos recuperados
        }
        
    except Exception as e:
        raise Exception(f"Error al comparar modelos: {str(e)}")