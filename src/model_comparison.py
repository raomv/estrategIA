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
import time  # ✅ AÑADIR IMPORT QUE FALTA
from llama_index.llms.ollama import Ollama
from cache_manager import get_cache_manager
from rag import RAG

# ✅ AÑADIR ESTAS DOS LÍNEAS:
import nest_asyncio
nest_asyncio.apply()

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
    include_retrieval_metrics: bool = False  # ✅ AÑADIR CAMPO QUE FALTA

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
        return [config.get("lllm_name", "deepseek-r1:1.5b")]

def academic_llamaindex_evaluation(request: CompareRequest, config: dict):
    """
    Evaluación académica usando LlamaIndex: Juez evalúa respuestas, no las genera.
    """
    start_time = time.time()  # ✅ AÑADIR ESTA LÍNEA
    
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
        
        # ✅ CORREGIDO: Crear evaluadores con manejo de errores individual
        evaluators = {}
        
        # Evaluadores que funcionan bien
        try:
            evaluators["faithfulness"] = FaithfulnessEvaluator(llm=judge_llm)
            print("✅ FaithfulnessEvaluator creado")
        except Exception as e:
            print(f"❌ Error creando FaithfulnessEvaluator: {e}")
        
        try:
            evaluators["relevancy"] = RelevancyEvaluator(llm=judge_llm)
            print("✅ RelevancyEvaluator creado")
        except Exception as e:
            print(f"❌ Error creando RelevancyEvaluator: {e}")
        
        # CorrectnessEvaluator necesita manejo especial
        try:
            evaluators["correctness"] = CorrectnessEvaluator(llm=judge_llm)
            print("✅ CorrectnessEvaluator creado")
        except Exception as e:
            print(f"⚠️ CorrectnessEvaluator no disponible: {e}")
        
        # Añadir SemanticSimilarityEvaluator (no requiere LLM)
        try:
            evaluators["semantic_similarity"] = SemanticSimilarityEvaluator()
            print("✅ SemanticSimilarityEvaluator creado")
        except Exception as e:
            print(f"❌ Error creando SemanticSimilarityEvaluator: {e}")
        
        # Añadir GuidelineEvaluator (con guidelines específicas)
        try:
            evaluators["guideline"] = GuidelineEvaluator(
                llm=judge_llm,
                guidelines="The response should be helpful, accurate, and based only on the provided context."
            )
            print("✅ GuidelineEvaluator creado")
        except Exception as e:
            print(f"⚠️ GuidelineEvaluator no disponible: {e}")
        
        print(f"📊 Evaluadores disponibles: {list(evaluators.keys())}")
        
        if not evaluators:
            return {"error": "No se pudieron crear evaluadores", "results": {}, "metrics": {}}
        
        # 3. Conectar al índice existente
        print(f"🔍 Conectando a colección existente: {collection_name}")
        
        temp_config = config.copy()
        temp_config["collection_name"] = collection_name
        
        initial_llm = Ollama(model=models_to_compare[0], url=config["llm_url"], request_timeout=300.0)
        rag_instance = RAG(config_file=temp_config, llm=initial_llm)
        shared_index = rag_instance.qdrant_index()
        
        if shared_index is None:
            raise ValueError(f"No se pudo conectar a la colección {collection_name}")
        
        print(f"✅ Conectado a índice existente: {collection_name}")
        
        results = {}
        metrics = {}
        
        # ✅ VARIABLE PARA QUERY_ENGINE (para retrieval metrics)
        query_engine = None
        
        # 4. Para cada modelo: evaluar con todos los evaluadores disponibles
        for model_name in models_to_compare:
            print(f"\n🔄 Procesando modelo: {model_name}")
            
            # ✅ INICIALIZAR model_metrics AQUÍ
            model_metrics = {}
            
            try:
                model_llm = Ollama(model=model_name, url=config["llm_url"], request_timeout=300.0)
                
                query_engine = shared_index.as_query_engine(
                    llm=model_llm,
                    similarity_top_k=config.get("similarity_top_k", 3),
                    response_mode="tree_summarize"
                )
                
                full_question = user_question + " You can only answer based on the provided context."
                response = query_engine.query(full_question)
                response_text = str(response).strip()
                
                print(f"   📝 Respuesta generada ({len(response_text)} chars): {response_text[:100]}...")
                results[model_name] = response_text
                
                # ✅ MANTENER extracción de contexts para debugging
                retrieved_contexts = []
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    for node in response.source_nodes:
                        retrieved_contexts.append(node.text)
                    print(f"      📄 Contexts recuperados del RAG: {len(retrieved_contexts)} fragmentos")
                    for i, ctx in enumerate(retrieved_contexts):
                        print(f"      📄 Context {i+1} ({len(ctx)} chars): {ctx[:150]}...")
                else:
                    print(f"      ⚠️ WARNING: No se recuperaron contexts de Qdrant")

                # ✅ CORREGIR evaluación según documentación LlamaIndex
                for metric_name, evaluator in evaluators.items():
                    try:
                        print(f"   📊 Evaluando {metric_name}...")
                        
                        if metric_name == "semantic_similarity":
                            eval_result = evaluator.evaluate_response(
                                response=response,
                                reference=user_question
                            )
                        else:
                            # faithfulness, relevancy, correctness, guideline
                            # NO pasar contexts - LlamaIndex los extrae automáticamente
                            eval_result = evaluator.evaluate_response(
                                query=user_question,
                                response=response
                            )
                        
                        print(f"      🔍 Resultado raw: {eval_result}")
                        print(f"      🔍 Tipo resultado: {type(eval_result)}")
                        
                        if hasattr(eval_result, '__dict__'):
                            print(f"      🔍 Atributos: {list(eval_result.__dict__.keys())}")
                        
                        # Extraer score
                        score = None
                        if hasattr(eval_result, 'score') and eval_result.score is not None:
                            score = float(eval_result.score)
                            print(f"      📏 Score encontrado: {score} (tipo: {type(eval_result.score)})")
                        else:
                            print(f"      ⚠️ Score no encontrado o es None")
                        
                        # Extraer passing
                        passing = None
                        if hasattr(eval_result, 'passing'):
                            passing = eval_result.passing
                            print(f"      ✅ Passing encontrado: {passing} (tipo: {type(passing)})")
                        
                        # Extraer feedback
                        feedback = ""
                        if hasattr(eval_result, 'feedback') and eval_result.feedback:
                            feedback = str(eval_result.feedback)
                            print(f"      💬 Feedback: {feedback[:100]}...")
                        
                        # Convertir score si es necesario
                        if score is None and passing is not None:
                            score = 1.0 if passing else 0.0
                            print(f"      🔄 Score convertido desde passing: {score}")
                        elif score is None:
                            score = 0.0
                            print(f"      ⚠️ Score por defecto: {score}")
                        
                        final_score = round(float(score), 2) if score is not None else 0.0
                        print(f"      ✅ Score convertido: {final_score}")
                        print(f"      ✅ {metric_name}: {final_score}")
                        
                        # ✅ GUARDAR EN model_metrics
                        model_metrics[metric_name] = final_score
                        
                    except Exception as e:
                        print(f"      ❌ Error en {metric_name}: {e}")
                        print(f"      🔍 Error completo: {type(e).__name__}: {e}")
                        model_metrics[metric_name] = 0.0
                
                # ✅ CALCULAR PUNTUACIÓN GENERAL CORRECTAMENTE
                valid_scores = [score for score in model_metrics.values() if isinstance(score, (int, float))]
                overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                model_metrics["overall_score"] = overall_score
                
                print(f"🎯 {model_name} - Puntuación general: {overall_score:.2f} (de {len(valid_scores)} métricas válidas)")
                
            except Exception as e:
                print(f"❌ Error procesando {model_name}: {str(e)}")
                results[model_name] = f"Error: {str(e)}"
                model_metrics = {"error": str(e)}
            
            # ✅ ASEGURAR QUE model_metrics SIEMPRE SE ASIGNA
            metrics[model_name] = model_metrics
        
        print("\n=== EVALUACIÓN ACADÉMICA COMPLETADA ===")
        print(f"📊 Contexto: Colección con pocos documentos puede resultar en scores altos")
        
        # ✅ EVALUACIÓN DE RETRIEVAL CORREGIDA
        retrieval_metrics = None
        if hasattr(request, 'include_retrieval_metrics') and request.include_retrieval_metrics and query_engine:
            print(f"\n🔍 === INICIANDO EVALUACIÓN DE RETRIEVAL ===")
            retrieval_metrics = evaluate_retrieval_metrics(
                query_engine=query_engine,
                user_query=request.message,  # ✅ CAMBIAR DE request.question a request.message
                config=config
            )
            print(f"🔍 === EVALUACIÓN DE RETRIEVAL COMPLETADA ===\n")
        
        return {
            "results": results,
            "metrics": metrics,
            "retrieval_metrics": retrieval_metrics,
            "metadata": {
                "judge_model": request.judge_model,
                "total_models": len(models_to_compare),
                "collection": request.collection,
                "evaluation_time": time.time() - start_time,
                "retrieval_evaluated": retrieval_metrics is not None
            }
        }
        
    except Exception as e:
        print(f"❌ Error en evaluación académica: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "results": {},
            "metrics": {},
            "retrieval_metrics": None,
            "metadata": {
                "judge_model": request.judge_model if hasattr(request, 'judge_model') else "unknown",
                "evaluation_time": time.time() - start_time if 'start_time' in locals() else 0
            }
        }

def evaluate_retrieval_metrics(query_engine, user_query, config):
    """
    Evalúa sistema de retrieval para la query específica del usuario
    Solo métricas nativas de LlamaIndex: Hit Rate y MRR
    """
    try:
        from llama_index.core.evaluation import RetrieverEvaluator
        
        print("🔍 === EVALUACIÓN DEL SISTEMA DE RETRIEVAL ===")
        print(f"📄 Modelo de Embeddings: {config.get('embedding_model', 'fastembed')}")
        print(f"🗄️ Vector Store: Qdrant")
        print(f"❓ Query del usuario: '{user_query[:50]}...'")
        
        # Extraer retriever del query_engine
        retriever = query_engine.retriever
        
        # ✅ Solo métricas nativas disponibles en LlamaIndex
        evaluator = RetrieverEvaluator.from_metric_names(
            ["hit_rate", "mrr"], 
            retriever=retriever
        )
        
        print(f"\n🔍 Evaluando retrieval para la query del usuario...")
        
        # Evaluar retrieval para la query específica
        result = evaluator.evaluate(query=user_query)
        
        print(f"   📊 Hit Rate: {result.hit_rate:.3f}")
        print(f"   📊 MRR: {result.mrr:.3f}")
        print(f"   📄 Documentos recuperados: {len(result.retrieved_nodes) if hasattr(result, 'retrieved_nodes') else 'N/A'}")
        
        # Interpretación para logs
        if result.hit_rate == 1.0:
            print(f"   ✅ Se encontraron documentos relevantes")
        else:
            print(f"   ⚠️ No se encontraron documentos suficientemente relevantes")
        
        return {
            "query": user_query,
            "hit_rate": result.hit_rate,
            "mrr": result.mrr,
            "retrieved_count": len(result.retrieved_nodes) if hasattr(result, 'retrieved_nodes') else 0,
            "interpretation": {
                "hit_rate_status": "success" if result.hit_rate == 1.0 else "warning",
                "mrr_quality": "excellent" if result.mrr > 0.8 else "good" if result.mrr > 0.5 else "poor"
            },
            "metadata": {
                "embedding_model": config.get("embedding_model", "fastembed"),
                "vector_store": "qdrant",
                "evaluation_timestamp": time.time()
            }
        }
        
    except Exception as e:
        print(f"❌ Error en evaluación de retrieval: {e}")
        return {
            "error": str(e),
            "query": user_query,
            "hit_rate": 0.0,
            "mrr": 0.0
        }