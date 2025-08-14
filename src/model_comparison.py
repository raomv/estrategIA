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
        
        # 4. Para cada modelo: evaluar con todos los evaluadores disponibles
        for model_name in models_to_compare:
            try:
                print(f"\n🔄 Procesando modelo: {model_name}")
                
                model_llm = Ollama(model=model_name, url=config["llm_url"], request_timeout=300.0)
                
                query_engine = shared_index.as_query_engine(
                    llm=model_llm,
                    similarity_top_k=config.get("similarity_top_k", 3),
                    response_mode="tree_summarize"
                )
                
                full_question = user_question + " You can only answer based on the provided context."
                response = query_engine.query(full_question)
                response_text = str(response).strip()
                
                # Extraer contexts de la respuesta RAG
                retrieved_contexts = []
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    for node in response.source_nodes:
                        retrieved_contexts.append(node.text)
                    
                    print(f"      📄 Contexts recuperados del RAG: {len(retrieved_contexts)} fragmentos")
                    for i, ctx in enumerate(retrieved_contexts):
                        print(f"      📄 Context {i+1} ({len(ctx)} chars): {ctx[:150]}...")
                else:
                    print(f"      ⚠️ WARNING: No se recuperaron contexts de Qdrant")
                
                # ✅ MEJORADO: Evaluación con manejo específico por tipo
                model_metrics = {}
                for metric_name, evaluator in evaluators.items():
                    try:
                        print(f"   📊 Evaluando {metric_name}...")
                        
                        # ✅ CORREGIDO: Manejo específico y correcto por evaluador
                        if metric_name == "semantic_similarity":
                            # SemanticSimilarityEvaluator no necesita contexts
                            eval_result = evaluator.evaluate_response(
                                response=response,
                                reference=user_question
                            )
                        elif metric_name == "correctness":
                            # CorrectnessEvaluator con contexts
                            eval_result = evaluator.evaluate_response(
                                query=user_question,
                                response=response,
                                contexts=retrieved_contexts,  # ← AÑADIR ESTA LÍNEA
                                reference="Expected accurate response based on provided context"
                            )
                        else:
                            # Todos los demás evaluadores (faithfulness, relevancy, guideline)
                            eval_result = evaluator.evaluate_response(
                                query=user_question,
                                response=response,
                                contexts=retrieved_contexts  # ← AÑADIR ESTA LÍNEA
                            )
                        
                        # ✅ DEBUGGING DETALLADO de la respuesta del evaluador
                        print(f"      🔍 Resultado raw: {eval_result}")
                        print(f"      🔍 Tipo resultado: {type(eval_result)}")
                        
                        if hasattr(eval_result, '__dict__'):
                            print(f"      🔍 Atributos: {list(eval_result.__dict__.keys())}")
                        
                        # Extraer score con debugging
                        score = None
                        passing = None
                        feedback = ""
                        
                        if hasattr(eval_result, 'score'):
                            score = eval_result.score
                            print(f"      📏 Score encontrado: {score} (tipo: {type(score)})")
                        
                        if hasattr(eval_result, 'passing'):
                            passing = eval_result.passing
                            print(f"      ✅ Passing encontrado: {passing} (tipo: {type(passing)})")
                        
                        if hasattr(eval_result, 'feedback'):
                            feedback = eval_result.feedback
                            print(f"      💬 Feedback: {str(feedback)[:200]}...")
                        
                        # Convertir a score numérico con debugging
                        if score is not None:
                            try:
                                numeric_score = float(score)
                                print(f"      ✅ Score convertido: {numeric_score}")
                            except (ValueError, TypeError) as e:
                                print(f"      ⚠️ Error convirtiendo score {score}: {e}")
                                numeric_score = 1.0 if passing else 0.0
                        elif passing is not None:
                            numeric_score = 1.0 if passing else 0.0
                            print(f"      🔄 Score desde passing: {numeric_score}")
                        else:
                            numeric_score = 0.0
                            print(f"      ⚠️ No score ni passing encontrado, usando 0.0")
                        
                        model_metrics[metric_name] = {
                            "score": numeric_score,
                            "passing": passing if passing is not None else (numeric_score > 0.5),
                            "feedback": str(feedback) if feedback else "Evaluación completada",
                            "raw_result": str(eval_result)  # Para debugging adicional
                        }
                        
                        print(f"      ✅ {metric_name}: {numeric_score:.2f}")
                        
                    except Exception as e:
                        print(f"      ❌ Error en {metric_name}: {str(e)}")
                        print(f"      🔍 Error completo: {type(e).__name__}: {e}")
                        model_metrics[metric_name] = {"error": str(e), "score": 0.0}
                
                # Calcular puntuación general
                valid_scores = [m["score"] for m in model_metrics.values() if "score" in m and "error" not in m]
                overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                model_metrics["overall_score"] = overall_score
                
                metrics[model_name] = model_metrics
                print(f"🎯 {model_name} - Puntuación general: {overall_score:.2f} (de {len(valid_scores)} métricas válidas)")
                
            except Exception as e:
                print(f"❌ Error procesando {model_name}: {str(e)}")
                results[model_name] = f"Error: {str(e)}"
                metrics[model_name] = {"error": str(e)}
        
        print("\n=== EVALUACIÓN ACADÉMICA COMPLETADA ===")
        print(f"📊 Contexto: Colección con pocos documentos puede resultar en scores altos")
        
        return {
            "results": results,
            "metrics": metrics,
            "judge_model": judge_model_name,
            "evaluation_method": "LlamaIndex Academic LLM-as-a-Judge (Full Metrics + Debug)",
            "academic_citation": "Liu et al. (2022) - LlamaIndex Framework + LLM-as-a-Judge Methodology"
        }
        
    except Exception as e:
        print(f"❌ Error en evaluación académica: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "results": {},
            "metrics": {},
            "judge_model": request.judge_model if hasattr(request, 'judge_model') else "unknown"
        }

# ❌ REMOVER todas las funciones relacionadas con RAGAS que estaban aquí anteriormente