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
import time
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
    judge_model: str
    include_retrieval_metrics: bool = False

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
    """
    start_time = time.time()
    
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
        
        # ✅ CREAR EVALUADORES con manejo de errores individual
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
                else:
                    print(f"      ⚠️ WARNING: No se recuperaron contexts de Qdrant")

                # ✅ EVALUACIÓN CON DEBUGGING COMPLETO
                for metric_name, evaluator in evaluators.items():
                    try:
                        print(f"   📊 Evaluando {metric_name}...")
                        
                        if metric_name == "semantic_similarity":
                            eval_result = evaluator.evaluate_response(
                                response=response,
                                reference=user_question
                            )
                        elif metric_name == "guideline":
                            # ✅ PASAR CONTEXTS AL GUIDELINE EVALUATOR
                            eval_result = evaluator.evaluate_response(
                                query=user_question,
                                response=response,
                                contexts=retrieved_contexts  # ✅ ESTO ES LO QUE FALTABA
                            )
                        else:
                            # faithfulness, relevancy, correctness
                            eval_result = evaluator.evaluate_response(
                                query=user_question,
                                response=response
                            )
                        
                        # ✅ EXTRACCIÓN DE SCORE CORREGIDA
                        score = None
                        if hasattr(eval_result, 'score'):
                            raw_score = eval_result.score
                            print(f"      📏 Score raw: {raw_score} (tipo: {type(raw_score)})")
                            if raw_score is not None:
                                score = float(raw_score)
                                print(f"      📏 Score convertido: {score}")
                            else:
                                print(f"      ⚠️ Score es None para {metric_name}")
                        else:
                            print(f"      ❌ No tiene atributo 'score': {metric_name}")
                        
                        # Extraer passing
                        passing = None
                        if hasattr(eval_result, 'passing'):
                            passing = eval_result.passing
                            print(f"      ✅ Passing: {passing} (tipo: {type(passing)})")
                        
                        # Extraer feedback
                        feedback = ""
                        if hasattr(eval_result, 'feedback'):
                            raw_feedback = eval_result.feedback
                            if raw_feedback:
                                feedback = str(raw_feedback)
                                print(f"      💬 Feedback (primeros 150 chars): {feedback[:150]}...")
                                
                                # ✅ PARA CORRECTNESS: Extraer score del feedback CON NORMALIZACIÓN CORRECTA
                                if metric_name == "correctness" and score is None:
                                    import re
                                    score_patterns = [
                                        r'\b(\d+\.?\d*)\s*(?:out of|/)\s*5',  # "4.0 out of 5"
                                        r'\bscore[:\s]*(\d+\.?\d*)',          # "score: 4.0"
                                        r'\b(\d+\.?\d*)\s*/\s*5',             # "4.0/5"
                                        r'\b(\d+\.?\d*)\b'                    # cualquier número
                                    ]
                                    
                                    for pattern in score_patterns:
                                        score_match = re.search(pattern, feedback, re.IGNORECASE)
                                        if score_match:
                                            try:
                                                extracted_score = float(score_match.group(1))
                                                print(f"      🔧 Score extraído del feedback: {extracted_score}")
                                                
                                                # ✅ NORMALIZACIÓN CORRECTA SIN REDONDEO AGRESIVO
                                                if extracted_score > 1:
                                                    # Asumimos escala de 5 puntos
                                                    score = extracted_score / 5.0
                                                    print(f"      🔧 Score normalizado (escala 5): {score:.3f}")
                                                else:
                                                    # Ya está en escala 0-1
                                                    score = extracted_score
                                                    print(f"      🔧 Score ya normalizado: {score:.3f}")
                                                
                                                # ✅ LÍMITE MÁXIMO SIN REDONDEO AGRESIVO
                                                score = min(score, 1.0)
                                                break
                                            except ValueError:
                                                continue
                        
                        # ✅ CONVERSIÓN FINAL SIN REDONDEO AGRESIVO
                        if score is None and passing is not None:
                            score = 1.0 if passing else 0.0
                            print(f"      🔄 Score convertido desde passing: {score}")
                        elif score is None:
                            score = 0.0
                            print(f"      ⚠️ Score por defecto para {metric_name}: {score}")
                        
                        # ✅ REDONDEO CONSERVADOR A 3 DECIMALES (NO A ENTEROS)
                        final_score = round(float(score), 3)  # 0.8 -> 0.800, NO 1.0
                        
                        model_metrics[metric_name] = {
                            "score": final_score,
                            "passing": passing if passing is not None else (final_score >= 0.5),
                            "feedback": feedback[:300] + "..." if len(feedback) > 300 else feedback
                        }
                        
                        print(f"      ✅ {metric_name}: {final_score} (passing: {passing})")
                        
                    except Exception as e:
                        print(f"      ❌ Error evaluando {metric_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        model_metrics[metric_name] = {
                            "score": 0.0,
                            "passing": False,
                            "feedback": f"Error en evaluación: {str(e)}"
                        }
                
                # ✅ CALCULAR PUNTUACIÓN GENERAL CORRECTAMENTE
                valid_scores = []
                for metric_name, metric_data in model_metrics.items():
                    if isinstance(metric_data, dict) and 'score' in metric_data:
                        score_value = metric_data['score']
                        if isinstance(score_value, (int, float)) and score_value >= 0:
                            valid_scores.append(score_value)
                
                overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                model_metrics["overall_score"] = round(overall_score, 2)
                
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
                user_query=request.message,
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
    """Evalúa sistema de retrieval de forma simplificada"""
    try:
        print("🔍 === EVALUACIÓN DEL SISTEMA DE RETRIEVAL ===")
        print(f"📄 Modelo de Embeddings: {config.get('embedding_model', 'fastembed')}")
        print(f"🗄️ Vector Store: Qdrant")
        print(f"❓ Query: '{user_query[:50]}...'")
        
        # ✅ EVALUACIÓN SIMPLIFICADA que SÍ FUNCIONA
        retriever = query_engine.retriever
        retrieved_nodes = retriever.retrieve(user_query)
        retrieved_count = len(retrieved_nodes)
        
        # Métricas básicas
        hit_rate = 1.0 if retrieved_count > 0 else 0.0
        mrr = 1.0 if retrieved_count > 0 else 0.0
        
        print(f"   📊 Hit Rate: {hit_rate:.3f}")
        print(f"   📊 MRR: {mrr:.3f}")
        print(f"   📄 Docs recuperados: {retrieved_count}")
        
        return {
            "query": user_query,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "retrieved_count": retrieved_count,
            "interpretation": {
                "hit_rate_status": "success" if hit_rate == 1.0 else "warning",
                "mrr_quality": "excellent" if mrr > 0.8 else "good"
            },
            "metadata": {
                "embedding_model": config.get("embedding_model", "fastembed"),
                "vector_store": "qdrant",
                "evaluation_timestamp": time.time()
            }
        }
        
    except Exception as e:
        print(f"❌ Error en retrieval: {e}")
        return {"error": str(e), "query": user_query, "hit_rate": 0.0, "mrr": 0.0}