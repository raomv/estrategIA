from fastapi import HTTPException
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from typing import Dict, List, Optional
import yaml
import ollama
import pandas as pd
import requests
import traceback

# Importar tu clase RAG y cache_manager
from rag import RAG
from cache_manager import get_cache_manager # Importar get_cache_manager

# Importaciones RAGAS siguiendo la documentación oficial
try:
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        # ContextRecall, # ContextRecall también necesita ground_truth y puede ser intensivo
    )
    from ragas.llms import LlamaIndexLLMWrapper
    from ragas.integrations.llama_index import evaluate
    from ragas import EvaluationDataset
    RAGAS_AVAILABLE = True
    print("✅ RAGAS importado correctamente con integración LlamaIndex")
except ImportError as e:
    print(f"❌ RAGAS no está disponible: {e}")
    RAGAS_AVAILABLE = False

# Importar LlamaIndex Settings
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
    judge_model: str  # ← Nuevo campo obligatorio

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
                print(f"✅ Modelos encontrados (directo): {available_models}")
                return available_models if available_models else [config.get("llm_name", "deepseek-r1:1.5b")]
            raise Exception("No se encontró 'models' en la respuesta directa de API")
    except Exception as e:
        print(f"❌ Error obteniendo modelos de Ollama: {e}")
        return [config.get("llm_name", "deepseek-r1:1.5b")]


def compare_models(request: CompareRequest, config_file: str, prompt_suffix: str):
    """
    Compara respuestas de múltiples modelos usando RAGAS oficial.
    Cada modelo seleccionado por el usuario recibe la misma pregunta y RAGAS evalúa cada respuesta independientemente,
    utilizando el propio modelo evaluado como juez para sus métricas.
    """
    try:
        print(f"=== INICIANDO COMPARACIÓN RAGAS OFICIAL ===")
        print(f"Consulta: {request.message[:100]}...")
        
        with open(config_file, "r") as conf:
            config = yaml.safe_load(conf)
        
        # CAMBIO: Solo usar los modelos que el usuario seleccionó explícitamente
        if not request.models or len(request.models) == 0:
            raise ValueError("No se han seleccionado modelos para comparar. El usuario debe seleccionar al menos un modelo desde la interfaz web.")
        
        models_to_compare = request.models
        print(f"Modelos seleccionados por el usuario: {models_to_compare}")
        print(f"RAGAS disponible: {RAGAS_AVAILABLE}")
        
        # Validar que los modelos seleccionados estén disponibles en Ollama
        available_models = get_available_models(config)
        invalid_models = [model for model in models_to_compare if model not in available_models]
        if invalid_models:
            print(f"⚠️ Modelos no disponibles en Ollama: {invalid_models}")
            print(f"✅ Modelos disponibles: {available_models}")
            # Filtrar solo los modelos válidos
            models_to_compare = [model for model in models_to_compare if model in available_models]
            if not models_to_compare:
                raise ValueError(f"Ninguno de los modelos seleccionados está disponible en Ollama. Modelos disponibles: {available_models}")
        
        print(f"Modelos válidos a comparar: {models_to_compare}")
        
        if not RAGAS_AVAILABLE:
            print("⚠️ RAGAS no disponible.")
            return generate_responses_only(models_to_compare, request.message, config, prompt_suffix)

        # 1. Asegurar y configurar el modelo de embeddings global UNA VEZ.
        print("🔧 Configurando modelo de embeddings local...")
        
        # Configurar explícitamente el embed_model ANTES de verificarlo
        cache_manager.ensure_embedding_model_ready(config)
        
        # Obtener el modelo de embeddings del cache manager y asignarlo explícitamente
        embed_model = cache_manager.get_cached_embedding_model()
        if embed_model is None:
            print("⚠️ Modelo de embeddings no encontrado en cache, creando nuevo...")
            embed_model = cache_manager.create_embedding_model(config)
        
        # Configurar LlamaSettings explícitamente 
        LlamaSettings.embed_model = embed_model
        print(f"✅ Modelo de embeddings configurado: {embed_model.model_name}")

        # Ya no se crea un LLM evaluador global aquí. Se creará uno por cada modelo a evaluar.
        
        results = {}
        metrics = {}
        
        # CAMBIO: Verificar si LlamaSettings.llm existe antes de acceder
        try:
            original_global_llm = LlamaSettings.llm
        except ValueError:
            # Si no hay LLM configurado, establecer None
            original_global_llm = None
            # Configurar un LLM temporal para evitar errores
            first_model = models_to_compare[0]
            LlamaSettings.llm = Ollama(model=first_model, url=config["llm_url"])
        
        for model_name in models_to_compare:
            try:
                print(f"📊 Iniciando evaluación para {model_name} con RAGAS...")
                
                # Configurar el LLM específico para este modelo en LlamaSettings y para RAGAS
                specific_llm_base = Ollama(model=model_name, url=config["llm_url"], request_timeout=config.get("llm_request_timeout", 300.0))
                LlamaSettings.llm = specific_llm_base # Establecer como LLM global temporalmente para el query_engine
                print(f"   Temporarily set LlamaSettings.llm to: {model_name}")

                # Crear el LLM wrapper para RAGAS usando el modelo actual
                ragas_llm_for_model = LlamaIndexLLMWrapper(specific_llm_base)
                print(f"   RAGAS LLM for {model_name} created.")

                # Inicializar métricas RAGAS con el LLM específico del modelo
                metrics_instances = [
                    Faithfulness(llm=ragas_llm_for_model),
                    AnswerRelevancy(llm=ragas_llm_for_model),
                    ContextPrecision(llm=ragas_llm_for_model),
                ]
                print(f"   Métricas RAGAS inicializadas para {model_name}")

                # Crear una instancia de RAG. Usará LlamaSettings.embed_model (global) y LlamaSettings.llm (específico del modelo).
                temp_rag_config = config.copy() # Para la colección correcta
                rag_for_model = RAG(config_file=temp_rag_config, llm=specific_llm_base) # Pasar LLM explícitamente
                
                # qdrant_index() usa el self.embed_model de RAG, que se inicializa desde LlamaSettings o cache_manager
                index_for_model = rag_for_model.qdrant_index() 

                if index_for_model is None:
                    raise ValueError(f"VectorStoreIndex es None para {model_name}. Qdrant podría no estar accesible o la colección no existe.")

                # as_query_engine() usará el LlamaSettings.llm y LlamaSettings.embed_model actuales.
                query_engine = index_for_model.as_query_engine(
                    similarity_top_k=config.get("similarity_top_k_ragas", 3), # Config específica para RAGAS
                    response_mode=config.get("response_mode_ragas", "tree_summarize")
                )

                if query_engine is None:
                    raise ValueError(f"QueryEngine es None para {model_name}")
                
                # Verificar componentes del query_engine
                if not hasattr(query_engine, '_retriever') or query_engine._retriever is None:
                    raise ValueError(f"Query engine para {model_name} no tiene un retriever válido.")
                if not hasattr(query_engine._retriever, 'retrieve'):
                    raise ValueError(f"Retriever para {model_name} no tiene método 'retrieve'.")

                print(f"   Query Engine para {model_name} creado. LLM: {query_engine._llm.model if hasattr(query_engine, '_llm') else 'N/A'}. Embeddings: {LlamaSettings.embed_model.model_name}")
                
                print(f"📋 Creando dataset RAGAS para {model_name}...")
                full_question = request.message + prompt_suffix
                ragas_data_list = [{"user_input": full_question, "ground_truth": f"Placeholder GT for {request.message}"}]
                
                dataset = EvaluationDataset.from_dict(ragas_data_list)
                print(f"   Dataset RAGAS creado para {model_name}.")
                
                print(f"🔬 Ejecutando RAGAS.evaluate para {model_name}...")
                evaluation_result = evaluate(
                    query_engine=query_engine,
                    metrics=metrics_instances,
                    dataset=dataset
                )
                print(f"   Evaluación RAGAS completada para {model_name}.")
                
                df = evaluation_result.to_pandas()
                if len(df) > 0:
                    row = df.iloc[0]
                    answer = str(row.get('answer', ''))
                    if not answer: # Si RAGAS no extrajo la respuesta, generarla.
                        print(f"   RAGAS no devolvió 'answer' para {model_name}, generando manualmente...")
                        response_obj = query_engine.query(full_question)
                        answer = str(response_obj).strip()
                    results[model_name] = answer
                    
                    faithfulness_score = row.get('faithfulness')
                    answer_relevancy_score = row.get('answer_relevancy') 
                    context_precision_score = row.get('context_precision')
                    
                    valid_scores = [s for s in [faithfulness_score, answer_relevancy_score, context_precision_score] if s is not None and not pd.isna(s) and isinstance(s, (float, int))]
                    overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
                    
                    metrics[model_name] = {
                        "faithfulness": float(faithfulness_score) if faithfulness_score is not None and not pd.isna(faithfulness_score) else None,
                        "answer_relevancy": float(answer_relevancy_score) if answer_relevancy_score is not None and not pd.isna(answer_relevancy_score) else None,
                        "context_relevancy": float(context_precision_score) if context_precision_score is not None and not pd.isna(context_precision_score) else None, # Mapeado desde ContextPrecision
                        "overall_score": overall_score
                    }
                    print(f"✅ Métricas para {model_name}: {metrics[model_name]}")
                else:
                    print(f"⚠️ No se obtuvieron filas de RAGAS para {model_name}")
                    results[model_name] = generate_fallback_response(model_name, request.message, config, prompt_suffix)
                    metrics[model_name] = {"error": "RAGAS no devolvió resultados en DataFrame"}

            except Exception as e:
                print(f"❌ Error evaluando {model_name} con RAGAS: {str(e)}")
                traceback.print_exc()
                results[model_name] = generate_fallback_response(model_name, request.message, config, prompt_suffix)
                metrics[model_name] = {"error": f"Excepción en RAGAS para {model_name}: {str(e)}"}
        
        LlamaSettings.llm = original_global_llm # Restaurar LLM global
        print("=== COMPARACIÓN COMPLETADA ===")
        return {"results": results, "metrics": metrics, "models_compared": models_to_compare, "ragas_available": RAGAS_AVAILABLE}
        
    except Exception as e:
        print(f"❌ Error general en compare_models: {str(e)}")
        traceback.print_exc()
        # Devolver un error estructurado que el frontend pueda manejar si es necesario
        return {
            "results": {}, 
            "metrics": {}, 
            "models_compared": request.models if request.models else [], 
            "ragas_available": RAGAS_AVAILABLE,
            "error": f"Error general en comparación: {str(e)}"
        }

def create_model_query_engine_for_fallback(model_name: str, config: Dict):
    """Crea un query engine simplificado para fallback, asegurando que LlamaSettings estén correctas."""
    try:
        # Asegurar que el embed_model esté configurado
        cache_manager.ensure_embedding_model_ready(config)
        embed_model = cache_manager.get_cached_embedding_model()
        if embed_model is None:
            embed_model = cache_manager.create_embedding_model(config)
        
        # Configurar explícitamente el embed_model
        LlamaSettings.embed_model = embed_model
        
        llm_instance = Ollama(model=model_name, url=config["llm_url"], request_timeout=config.get("llm_request_timeout", 300.0))
        
        # Guardar y restaurar LLM global para esta operación específica
        original_llm = getattr(LlamaSettings, 'llm', None)
        LlamaSettings.llm = llm_instance
        
        temp_rag_config = config.copy()
        rag_instance = RAG(config_file=temp_rag_config, llm=llm_instance) # RAG usa LlamaSettings
        index_instance = rag_instance.qdrant_index()

        if index_instance is None:
            raise ValueError("Fallback: VectorStoreIndex es None.")
            
        query_engine = index_instance.as_query_engine() # Usa LlamaSettings.llm
        
        LlamaSettings.llm = original_llm # Restaurar
        
        if query_engine is None:
            raise ValueError("Fallback: QueryEngine es None.")
        return query_engine
    except Exception as e:
        print(f"❌ Error creando query engine de fallback para {model_name}: {e}")
        return None

def generate_fallback_response(model_name: str, message: str, config: Dict, prompt_suffix: str) -> str:
    print(f"⚠️ Generando respuesta de fallback para {model_name}...")
    try:
        query_engine = create_model_query_engine_for_fallback(model_name, config)
        if query_engine:
            response = query_engine.query(message + prompt_suffix)
            return str(response).strip()
        return f"Error: No se pudo crear query engine de fallback para {model_name}."
    except Exception as e:
        print(f"❌ Error generando respuesta de fallback para {model_name}: {e}")
        return f"Error al generar respuesta para {model_name}: {str(e)}"

def generate_responses_only(models_to_compare: List[str], message: str, config: Dict, prompt_suffix: str):
    print("🔄 Generando solo respuestas (RAGAS no disponible o falló globalmente)")
    results = {}
    for model_name in models_to_compare:
        results[model_name] = generate_fallback_response(model_name, message, config, prompt_suffix)
    return {
        "results": results, "metrics": {}, "models_compared": models_to_compare,
        "ragas_available": False, "error": "RAGAS no disponible o falló la evaluación."
    }

def academic_llamaindex_evaluation(request: CompareRequest, config: dict):
    """
    Evaluación académica usando LlamaIndex: Juez evalúa respuestas, no las genera.
    Basado en metodología LLM-as-a-Judge académicamente validada.
    """
    try:
        print("=== EVALUACIÓN ACADÉMICA CON LLAMAINDEX ===")
        
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
        
        # 2. Crear el LLM juez (solo para evaluar, no para responder)
        judge_llm = Ollama(model=judge_model_name, url=config["llm_url"], request_timeout=300.0)
        print(f"🏅 Juez configurado: {judge_model_name}")
        
        # 3. Crear TODOS los evaluadores de LlamaIndex disponibles
        evaluators = {
            "faithfulness": FaithfulnessEvaluator(llm=judge_llm),
            "relevancy": RelevancyEvaluator(llm=judge_llm),
            "correctness": CorrectnessEvaluator(llm=judge_llm),
            "semantic_similarity": SemanticSimilarityEvaluator(llm=judge_llm),
        }
        print(f"📊 Evaluadores creados: {list(evaluators.keys())}")
        
        results = {}
        metrics = {}
        
        # 4. Para cada modelo: generar respuesta y evaluarla con el juez
        for model_name in models_to_compare:
            try:
                print(f"\n🔄 Procesando modelo: {model_name}")
                
                # Crear RAG para el modelo específico
                temp_config = config.copy()
                temp_config["collection_name"] = collection_name
                
                model_llm = Ollama(model=model_name, url=config["llm_url"], request_timeout=300.0)
                LlamaSettings.llm = model_llm
                
                # Crear RAG instance
                from rag import RAG
                rag_instance = RAG(config_file=temp_config, llm=model_llm)
                index = rag_instance.qdrant_index()
                
                if index is None:
                    raise ValueError(f"No se pudo crear índice para {model_name}")
                
                query_engine = index.as_query_engine(
                    similarity_top_k=config.get("similarity_top_k", 3),
                    response_mode="tree_summarize"
                )
                
                # Generar respuesta del modelo
                response = query_engine.query(user_question + " You can only answer based on the provided context.")
                results[model_name] = str(response).strip()
                print(f"✅ Respuesta generada para {model_name}")
                
                # El juez evalúa esta respuesta usando TODAS las métricas
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
                            "passing": eval_result.passing,
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