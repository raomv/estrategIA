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
from ragas_integration import calculate_ragas_metrics  # NUEVO IMPORT

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
    include_ragas_metrics: bool = False 

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

# Y que la función use CompareRequest:
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
            evaluators["faithfulness"] = FaithfulnessEvaluator(llm=judge_llm)  # ✅ SIN prompt_template
            print("✅ FaithfulnessEvaluator creado")
        except Exception as e:
            print(f"❌ Error creando FaithfulnessEvaluator: {e}")
        
        try:
            evaluators["relevancy"] = RelevancyEvaluator(llm=judge_llm)  # ✅ SIN prompt_template
            print("✅ RelevancyEvaluator creado")
        except Exception as e:
            print(f"❌ Error creando RelevancyEvaluator: {e}")
        
        try:
            evaluators["correctness"] = CorrectnessEvaluator(llm=judge_llm)  # ✅ SIN prompt_template
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
                guidelines="IGNORE your training knowledge. Evaluate ONLY against the provided context. The response should use only information from the given context documents. Do not apply external knowledge about typical military structures or common practices."
            )
            print("✅ GuidelineEvaluator creado (con instrucciones estrictas)")
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
                            # ✅ NO PASAR CONTEXTS - GuidelineEvaluator los extrae automáticamente
                            eval_result = evaluator.evaluate_response(
                                query=user_question,
                                response=response
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
                                print(f"      💬 Feedback completo: {feedback}")  # ✅ CAMBIO 1: SIN CORTE
                                
                                # ✅ PARA CORRECTNESS: Extraer score del feedback CON NORMALIZACIÓN CORRECTA
                                if metric_name == "correctness" and score is None:
                                    import re
                                    
                                    # ✅ BUSCAR PRIMERO DENTRO DE <think> TAGS
                                    think_content = ""
                                    think_match = re.search(r'<think>(.*?)</think>', feedback, flags=re.DOTALL)
                                    if think_match:
                                        think_content = think_match.group(1)
                                        print(f"      🧠 Contenido <think>: {think_content}")
                                    
                                    score_patterns = [
                                        r'\b(\d+\.?\d*)\s*(?:out of|/)\s*5',  # "4.0 out of 5"
                                        r'\bscore[:\s]*(\d+\.?\d*)',          # "score: 4.0"
                                        r'\b(\d+\.?\d*)\s*/\s*5',             # "4.0/5"
                                        r'\b(\d+\.?\d*)\b'                    # cualquier número
                                    ]
                                    
                                    # ✅ BUSCAR PRIMERO EN <think>, LUEGO EN FEEDBACK COMPLETO
                                    search_areas = [think_content, feedback] if think_content else [feedback]
                                    
                                    for search_text in search_areas:
                                        for pattern in score_patterns:
                                            score_match = re.search(pattern, search_text, re.IGNORECASE)
                                            if score_match:
                                                try:
                                                    extracted_score = float(score_match.group(1))
                                                    print(f"      🔧 Score extraído de {'<think>' if search_text == think_content else 'feedback'}: {extracted_score}")
                                                    
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
                                        if score is not None:
                                            break
                        
                        # ✅ CONVERSIÓN FINAL CON NORMALIZACIÓN UNIVERSAL
                        if score is None and passing is not None:
                            score = 1.0 if passing else 0.0
                            print(f"      🔄 Score convertido desde passing: {score}")
                        elif score is None:
                            score = 0.0
                            print(f"      ⚠️ Score por defecto para {metric_name}: {score}")
                        
                        # ✅ NORMALIZACIÓN A ESCALA 0-1 PARA TODOS
                        if score > 1.0:
                            # Si el score viene en escala 0-5 o 0-10, normalizar
                            if score <= 5.0:
                                score = score / 5.0  # Escala 0-5 → 0-1
                                print(f"      🔧 Score normalizado (escala 5): {score:.3f}")
                            elif score <= 10.0:
                                score = score / 10.0  # Escala 0-10 → 0-1
                                print(f"      🔧 Score normalizado (escala 10): {score:.3f}")
                            else:
                                score = 1.0  # Clamp a máximo 1.0
                                print(f"      ⚠️ Score clampeado a 1.0: {score}")
                        
                        # ✅ GARANTIZAR RANGO 0-1
                        score = max(0.0, min(1.0, score))
                        final_score = round(float(score), 3)
                        
                        # ✅ CAMBIO 2: ELIMINAR 'passing' DEL JSON AL FRONTEND
                        model_metrics[metric_name] = {
                            "score": final_score,
                            "feedback": feedback  # ✅ SIN CORTE [:300]
                        }
                        
                        print(f"      ✅ {metric_name}: {final_score}")
                        
                    except Exception as e:
                        print(f"      ❌ Error evaluando {metric_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        model_metrics[metric_name] = {
                            "score": 0.0,
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
        
        # ✅ ÚNICA MODIFICACIÓN - AÑADIR MÉTRICAS RAGAS
        try:
            if config.get("include_ragas_metrics", False):
                logger.info("Calculating RAGAS metrics...")
                
                # Obtener mejor respuesta como ground truth
                best_response_text = ""
                if responses:
                    # Usar la primera respuesta como referencia o la mejor puntuada
                    first_model = list(responses.keys())[0]
                    if hasattr(responses[first_model]['response'], 'response'):
                        best_response_text = responses[first_model]['response'].response
                    else:
                        best_response_text = str(responses[first_model]['response'])
                
                # Calcular métricas RAGAS
                ragas_metrics = calculate_ragas_metrics(
                    user_query=user_query,
                    model_responses=responses,
                    contexts=contexts if contexts else [],
                    judge_response=best_response_text
                )
                
                # Añadir métricas RAGAS a resultados existentes
                for model_name, ragas_data in ragas_metrics.items():
                    if model_name in responses:
                        responses[model_name].update(ragas_data)
                        logger.info(f"Added RAGAS metrics to {model_name}: {ragas_data}")
                
        except Exception as e:
            logger.error(f"RAGAS evaluation failed, continuing without RAGAS metrics: {e}")
            # Si RAGAS falla, el flujo continúa normalmente
            pass
        
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
    """Evalúa retrieval con métricas label-free adaptado al proyecto"""
    import time
    import numpy as np
    from itertools import combinations
    
    try:
        print("🔍 === EVALUACIÓN DE RETRIEVAL (LABEL-FREE) ===")
        
        # ✅ RETRIEVAL con timing
        retriever = query_engine.retriever
        t0 = time.time()
        nodes = retriever.retrieve(user_query)
        t1 = time.time()
        retrieval_time_ms = round((t1 - t0) * 1000.0, 2)
        
        k = len(nodes)
        print(f"📄 Documentos recuperados: {k} en {retrieval_time_ms}ms")
        
        if k == 0:
            return {
                "query": user_query,
                "retrieved_count": 0,
                "retrieval_time_ms": retrieval_time_ms,
                "error": "No se recuperaron documentos"
            }
        
        # ✅ EXTRAER SCORES (compatibility con tu setup)
        scores = []
        for i, node in enumerate(nodes):
            score = getattr(node, 'score', 0.0)
            print(f"   📄 Doc {i+1}: score={score:.4f}")
            scores.append(float(score))
        
        # ✅ MÉTRICAS BASADAS EN SCORES
        scores_sorted = sorted(scores, reverse=True)
        score_at_1 = scores_sorted[0]
        mean_score = float(np.mean(scores))
        var_score = float(np.var(scores)) if k > 1 else 0.0
        margin_at_1 = float(scores_sorted[0] - scores_sorted[1]) if k > 1 else score_at_1
        
        threshold = float(config.get("similarity_threshold", 0.7))
        accept_rate = float(np.mean([s >= threshold for s in scores]))
        
        print(f"📊 Score@1: {score_at_1:.4f}")
        print(f"📊 Mean Score: {mean_score:.4f}")
        print(f"📊 Accept Rate@{threshold}: {accept_rate:.4f}")
        
        # ✅ MÉTRICAS DE EMBEDDINGS (adaptado a tu proyecto)
        try:
            # Acceder al embed_model desde el query_engine
            embed_model = None
            if hasattr(query_engine, '_service_context') and hasattr(query_engine._service_context, 'embed_model'):
                embed_model = query_engine._service_context.embed_model
            elif hasattr(query_engine, 'retriever') and hasattr(query_engine.retriever, '_embed_model'):
                embed_model = query_engine.retriever._embed_model
            
            if embed_model:
                # Query embedding
                q_emb = np.array(embed_model.get_query_embedding(user_query), dtype=np.float32)
                
                # Doc embeddings - intentar diferentes métodos
                doc_embs = []
                for node in nodes:
                    try:
                        # Método 1: embedding directo en node
                        if hasattr(node, 'embedding') and node.embedding is not None:
                            doc_embs.append(np.array(node.embedding, dtype=np.float32))
                        # Método 2: desde node.node
                        elif hasattr(node, 'node') and hasattr(node.node, 'embedding') and node.node.embedding:
                            doc_embs.append(np.array(node.node.embedding, dtype=np.float32))
                        # Método 3: regenerar embedding del texto
                        else:
                            text = getattr(node, 'text', getattr(node, 'content', ''))
                            if hasattr(node, 'node'):
                                text = getattr(node.node, 'text', text)
                            if text:
                                emb = embed_model.get_text_embedding(text)
                                doc_embs.append(np.array(emb, dtype=np.float32))
                    except Exception as e:
                        print(f"⚠️ Error obteniendo embedding para doc: {e}")
                        continue
                
                if doc_embs and len(doc_embs) == k:
                    # Query-Doc similarities
                    qd_sims = []
                    for doc_emb in doc_embs:
                        sim = np.dot(q_emb, doc_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb))
                        qd_sims.append(float(sim))
                    
                    qd_mean = float(np.mean(qd_sims))
                    qd_max = float(np.max(qd_sims))
                    
                    # Doc-Doc coherence
                    if k > 1:
                        dd_sims = []
                        for i in range(k):
                            for j in range(i+1, k):
                                sim = np.dot(doc_embs[i], doc_embs[j]) / (
                                    np.linalg.norm(doc_embs[i]) * np.linalg.norm(doc_embs[j])
                                )
                                dd_sims.append(float(sim))
                        docdoc_coherence = float(np.mean(dd_sims))
                        diversity = float(1.0 - docdoc_coherence)
                    else:
                        docdoc_coherence = 1.0
                        diversity = 0.0
                    
                    print(f"📊 Query-Doc Mean: {qd_mean:.4f}")
                    print(f"📊 Doc-Doc Coherence: {docdoc_coherence:.4f}")
                    print(f"📊 Diversity: {diversity:.4f}")
                    
                else:
                    print("⚠️ No se pudieron obtener embeddings para métricas geométricas")
                    qd_mean = qd_max = docdoc_coherence = diversity = None
            else:
                print("⚠️ No se pudo acceder al embed_model")
                qd_mean = qd_max = docdoc_coherence = diversity = None
                
        except Exception as e:
            print(f"⚠️ Error calculando métricas de embeddings: {e}")
            qd_mean = qd_max = docdoc_coherence = diversity = None
        
        # ✅ MÉTRICAS OPERACIONALES
        unique_sources = len(set(
            getattr(node, 'doc_id', getattr(getattr(node, 'node', node), 'doc_id', f'doc_{i}'))
            for i, node in enumerate(nodes)
        ))
        
        result = {
            "query": user_query,
            "retrieved_count": k,
            "retrieval_time_ms": retrieval_time_ms,
            "score_at_1": round(score_at_1, 4),
            "mean_score": round(mean_score, 4),
            "var_score": round(var_score, 6),
            "margin_at_1": round(margin_at_1, 4),
            "accept_rate_at_threshold": round(accept_rate, 4),
            "threshold_used": threshold,
            "unique_sources": unique_sources,
            "metadata": {
                "embedding_model": config.get("embedding_model", "fastembed"),
                "vector_store": "qdrant", 
                "evaluation_timestamp": time.time(),
                "similarity_top_k": config.get("similarity_top_k", 3)
            }
        }
        
        # Añadir métricas de embeddings si están disponibles
        if qd_mean is not None:
            result.update({
                "qd_mean": round(qd_mean, 4),
                "qd_max": round(qd_max, 4),
                "docdoc_coherence": round(docdoc_coherence, 4),
                "diversity": round(diversity, 4)
            })
        
        return result
        
    except Exception as e:
        print(f"❌ Error en evaluate_retrieval_metrics: {e}")
        return {
            "query": user_query,
            "error": str(e),
            "retrieved_count": 0
        }