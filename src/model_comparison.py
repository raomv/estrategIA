from fastapi import HTTPException
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from typing import Dict, List, Optional
import yaml
import ollama
import requests
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import statistics
from llama_index.llms.ollama import Ollama
from cache_manager import get_cache_manager
from rag import RAG  # Añadir este import que falta

# Importar LlamaIndex Settings y Evaluators (SIN RAGAS)
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    GuidelineEvaluator
)

# Obtener instancia del cache_manager
cache_manager = get_cache_manager()

class CompareRequest(BaseModel):
    message: str  
    models: List[str]
    collection: str
    judge_model: str  # Campo obligatorio para el juez

def get_available_models(config):
    """Obtiene la lista de modelos disponibles en Ollama."""
    try:
        print("🔍 Consultando modelos disponibles en Ollama...")
        try:
            models_response = ollama.list()
            print(f"📋 Respuesta de ollama.list(): {models_response}")
            if isinstance(models_response, dict) and 'models' in models_response:
                available_models = [model.get('name') for model in models_response['models'] if model.get('name')]
                print(f"✅ Modelos encontrados: {available_models}")
                return available_models if available_models else [config.get("llm_name", "deepseek-r1:1.5b")]
            else:
                raise Exception("Formato de respuesta inválido de ollama.list()")
        except Exception as ollama_error:
            print(f"❌ Error con ollama.list(): {ollama_error}, intentando fallback...")
            response = requests.get(f"{config.get('llm_url', 'http://localhost:11434')}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'models' in data:
                available_models = [model.get('name') for model in data['models'] if model.get('name')]
                print(f"✅ Modelos encontrados via fallback: {available_models}")
                return available_models if available_models else [config.get("llm_name", "deepseek-r1:1.5b")]
            else:
                raise Exception("No se pudieron obtener los modelos disponibles")
                
    except Exception as e:
        print(f"❌ Error general obteniendo modelos: {e}")
        return [config.get("llm_name", "deepseek-r1:1.5b")]

def academic_llamaindex_evaluation(request: CompareRequest, config: dict):
    """
    Evaluación académica usando LlamaIndex: Juez evalúa respuestas, no las genera.
    Basado en metodología LLM-as-a-Judge académicamente validada.
    """
    try:
        print("=== EVALUACIÓN ACADÉMICA CON LLAMAINDEX NATIVO ===")
        
        models_to_compare = request.models
        judge_model_name = request.judge_model
        user_question = request.message
        collection_name = request.collection
        
        print(f"🎯 Modelos a evaluar: {models_to_compare}")
        print(f"👨‍⚖️ Modelo juez: {judge_model_name}")
        print(f"📋 Colección: {collection_name}")
        print(f"❓ Pregunta: {user_question[:100]}...")
        
        # Validar que juez no esté en modelos a comparar
        if judge_model_name in models_to_compare:
            raise ValueError("El modelo juez no puede estar en la lista de modelos a comparar")
        
        # 1. Configurar embeddings una vez
        cache_manager.ensure_embedding_model_ready(config)
        embed_model = cache_manager.get_cached_embedding_model()
        LlamaSettings.embed_model = embed_model
        
        # 2. Crear el LLM juez
        judge_llm = Ollama(model=judge_model_name, url=config["llm_url"], request_timeout=300.0)
        print(f"🏅 Juez configurado: {judge_model_name}")
        
        # 3. Crear evaluadores de LlamaIndex
        evaluators = {}
        try:
            evaluators["faithfulness"] = FaithfulnessEvaluator(llm=judge_llm)
            evaluators["relevancy"] = RelevancyEvaluator(llm=judge_llm)
            evaluators["correctness"] = CorrectnessEvaluator(llm=judge_llm)
            print("✅ Evaluadores creados: faithfulness, relevancy, correctness")
        except Exception as e:
            print(f"❌ Error creando evaluadores: {e}")
            return {"error": f"Error creando evaluadores: {e}", "results": {}, "metrics": {}}
        
        # ✅ SIMPLIFICADO: Conectar al índice existente sin chunk_size
        print(f"🔍 Conectando a colección existente: {collection_name}")
        
        # Crear configuración temporal para conexión
        temp_config = config.copy()
        temp_config["collection_name"] = collection_name
        
        # Usar el primer modelo para la conexión inicial al índice
        initial_llm = Ollama(model=models_to_compare[0], url=config["llm_url"], request_timeout=300.0)
        
        # Crear instancia RAG (ahora con import correcto)
        rag_instance = RAG(config_file=temp_config, llm=initial_llm)
        shared_index = rag_instance.qdrant_index()
        
        if shared_index is None:
            raise ValueError(f"No se pudo conectar a la colección {collection_name}")
        
        print(f"✅ Conectado a índice existente: {collection_name}")
        
        results = {}
        metrics = {}
        
        # 4. Para cada modelo: solo cambiar LLM, reutilizar índice
        for model_name in models_to_compare:
            try:
                print(f"\n🔄 Procesando modelo: {model_name}")
                
                # Crear LLM específico para este modelo
                model_llm = Ollama(model=model_name, url=config["llm_url"], request_timeout=300.0)
                
                # ✅ CLAVE: Reutilizar índice, cambiar solo LLM
                query_engine = shared_index.as_query_engine(
                    llm=model_llm,  # Solo cambiar el LLM
                    similarity_top_k=config.get("similarity_top_k", 3),
                    response_mode="tree_summarize"
                )
                
                # Generar respuesta del modelo
                full_question = user_question + " You can only answer based on the provided context."
                response = query_engine.query(full_question)
                results[model_name] = str(response).strip()
                print(f"✅ Respuesta generada para {model_name}")
                
                # El juez evalúa esta respuesta usando las métricas académicas
                model_metrics = {}
                for metric_name, evaluator in evaluators.items():
                    try:
                        print(f"   📊 Evaluando {metric_name}...")
                        
                        eval_result = evaluator.evaluate_response(
                            query=user_question, 
                            response=response
                        )
                        
                        model_metrics[metric_name] = {
                            "score": eval_result.score if hasattr(eval_result, 'score') else (1.0 if eval_result.passing else 0.0),
                            "passing": eval_result.passing if hasattr(eval_result, 'passing') else True,
                            "feedback": eval_result.feedback if hasattr(eval_result, 'feedback') else "Evaluación completada"
                        }
                        print(f"      ✅ {metric_name}: {model_metrics[metric_name]['score']:.2f}")
                        
                    except Exception as e:
                        print(f"      ❌ Error en {metric_name}: {str(e)}")
                        model_metrics[metric_name] = {"error": str(e), "score": 0.0}
                
                # Calcular puntuación general
                valid_scores = [m["score"] for m in model_metrics.values() if "score" in m and "error" not in m]
                overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                model_metrics["overall_score"] = overall_score
                
                metrics[model_name] = model_metrics
                print(f"🎯 {model_name} - Puntuación general: {overall_score:.2f}")
                
            except Exception as e:
                print(f"❌ Error procesando {model_name}: {str(e)}")
                results[model_name] = f"Error: {str(e)}"
                metrics[model_name] = {"error": str(e)}
        
        print("\n=== EVALUACIÓN ACADÉMICA COMPLETADA ===")
        
        return {
            "results": results,
            "metrics": metrics,
            "judge_model": judge_model_name,
            "evaluation_method": "LlamaIndex Academic LLM-as-a-Judge",
            "academic_citation": "Liu et al. (2022) - LlamaIndex Framework + LLM-as-a-Judge Methodology"
        }
        
    except Exception as e:
        print(f"❌ Error en evaluación académica: {str(e)}")
        return {
            "error": str(e),
            "results": {},
            "metrics": {},
            "judge_model": request.judge_model if hasattr(request, 'judge_model') else "unknown"
        }

# ❌ REMOVER todas las funciones relacionadas con RAGAS que estaban aquí anteriormente