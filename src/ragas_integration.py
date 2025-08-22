"""
Integración de métricas RAGAS para evaluación estándar.
RAGAS 0.2.0 con LlamaIndex (sin LangChain)
"""
import logging
import os
import traceback
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def calculate_ragas_metrics(user_query, model_responses, contexts, judge_response, config=None):
    """
    Calcula métricas RAGAS usando el MISMO modelo juez seleccionado por el usuario
    RAGAS 0.2.0 + LlamaIndex (sin LangChain)
    """
    try:
        # ✅ CONFIGURAR TIMEOUTS DE RAGAS ANTES DE TODO
        os.environ["RAGAS_EXECUTOR_TIMEOUT"] = "1800"  # 30 minutos
        os.environ["RAGAS_DO_NOT_TRACK"] = "true"
        
        # ✅ USAR EL MISMO PATRÓN QUE LAS MÉTRICAS NATIVAS
        judge_model_name = config.get("judge_model") if config else None
        llm_url = config.get("llm_url") if config else "http://localhost:11434"
        
        if not judge_model_name:
            print("❌ No se proporcionó modelo juez para RAGAS")
            return {}
        
        print(f"🎯 RAGAS usando modelo juez seleccionado: {judge_model_name}")
        
        # ✅ IMPORTS CORRECTOS PARA RAGAS 0.2.0 CON LLAMAINDEX
        try:
            from ragas.llms.base import LlamaIndexLLMWrapper
            from ragas.embeddings.base import LlamaIndexEmbeddingsWrapper
            print("✅ Imports correctos desde .base funcionando")
        except ImportError as e:
            print(f"❌ No se pudo importar desde .base: {e}")
            return {}
        
        # ✅ CREAR EL MISMO LLM QUE USAN LAS MÉTRICAS NATIVAS CON TIMEOUT EXTENDIDO
        from llama_index.llms.ollama import Ollama
        judge_llm = Ollama(model=judge_model_name, base_url=llm_url, request_timeout=1800.0)  # 30 minutos
        
        # ✅ CONFIGURAR RAGAS CON ESE LLM (SOLO LLAMAINDEX)
        from ragas import evaluate
        # ✅ IMPORTAR SOLO LAS MÉTRICAS QUE FUNCIONAN
        from ragas.metrics import faithfulness, context_recall  # ← SOLO ESTAS DOS

        # Usar los wrappers correctos
        ragas_llm = LlamaIndexLLMWrapper(llm=judge_llm)

        # ✅ USAR EL MISMO EMBED_MODEL QUE EL PROYECTO
        from cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        embed_model = cache_manager.get_cached_embedding_model()
        
        if embed_model is None:
            print("❌ No se pudo obtener el modelo de embeddings del cache")
            return {}
            
        ragas_embeddings = LlamaIndexEmbeddingsWrapper(embeddings=embed_model)

        # ✅ CONFIGURAR RUNCONFIG PARA RAGAS 0.2.0 SEGÚN DOCUMENTACIÓN OFICIAL
        from ragas.run_config import RunConfig

        # Crear configuración con timeouts extendidos según documentación
        ragas_run_config = RunConfig(
            timeout=1800,        # 30 minutos (vs 180s por defecto)
            max_retries=3,       # Reintentos en caso de fallo temporal
            max_wait=120,        # Máximo 2 minutos entre reintentos
            max_workers=1        # Un solo worker para evitar sobrecarga
        )

        print(f"🔧 RunConfig configurado: timeout={ragas_run_config.timeout}s, max_retries={ragas_run_config.max_retries}")

        # ✅ CONFIGURAR SOLO LAS MÉTRICAS QUE FUNCIONAN CON RUNCONFIG
        print(f"🔧 === CONFIGURANDO MÉTRICAS RAGAS CON RUNCONFIG ===")

        # Para faithfulness: necesita question, answer, contexts
        try:
            faithfulness.llm = ragas_llm
            print(f"✅ Faithfulness configurado con LLM")
        except Exception as e:
            print(f"❌ Error configurando faithfulness: {e}")
            return {}

        # Para context_recall: necesita question, ground_truth, contexts, embeddings
        try:
            context_recall.llm = ragas_llm
            context_recall.embeddings = ragas_embeddings
            print(f"✅ Context recall configurado con LLM y embeddings")
        except Exception as e:
            print(f"❌ Error configurando context_recall: {e}")
            return {}

        print(f"✅ RAGAS configurado con juez: {judge_model_name}")
        
        # Calcular métricas
        ragas_results = {}
        
        for model_name, response_text in model_responses.items():
            try:
                from datasets import Dataset
                
                # ✅ VALIDACIÓN EXHAUSTIVA DE DATOS DE ENTRADA
                print(f"\n🔍 === VALIDACIÓN EXHAUSTIVA PARA {model_name} ===")
                
                # Validar user_query
                if not user_query or len(user_query.strip()) < 5:
                    print(f"❌ User query inválido: '{user_query}'")
                    continue
                print(f"✅ User query válido: {len(user_query)} chars")
                
                # Validar response_text
                if not response_text or len(response_text.strip()) < 10:
                    print(f"❌ Response text inválido: '{response_text[:50]}...'")
                    continue
                print(f"✅ Response text válido: {len(response_text)} chars")
                
                # Validar contexts
                if not contexts or len(contexts) == 0:
                    print(f"❌ Contexts vacíos o inválidos")
                    continue
                
                valid_contexts = [ctx for ctx in contexts if ctx and len(ctx.strip()) > 10]
                if len(valid_contexts) == 0:
                    print(f"❌ No hay contexts válidos (>10 chars)")
                    continue
                print(f"✅ Contexts válidos: {len(valid_contexts)} de {len(contexts)}")
                
                # Validar judge_response
                if not judge_response or len(str(judge_response).strip()) < 20:
                    print(f"❌ Judge response inválido o muy corto")
                    ground_truth = f"A comprehensive answer to the question: {user_query}"
                    print(f"🔄 Usando ground truth generado: {ground_truth[:100]}...")
                else:
                    ground_truth = str(judge_response).strip()
                    print(f"✅ Judge response válido: {len(ground_truth)} chars")
                
                # ✅ PREPARAR DATOS SIN TRUNCAR DEMASIADO (RAGAS necesita contenido suficiente)
                print(f"🔄 === PREPARANDO DATOS PARA RAGAS ===")
                
                # Usar máximo 3 contextos más largos (500 chars cada uno)
                prepared_contexts = valid_contexts[:3]
                if len(prepared_contexts[0]) > 500:
                    prepared_contexts = [ctx[:500] + "..." for ctx in prepared_contexts]
                    
                # No truncar tanto la respuesta (máximo 400 chars)
                prepared_response = response_text
                if len(prepared_response) > 400:
                    prepared_response = response_text[:400] + "..."
                
                # Ground truth más largo si es posible (máximo 300 chars)
                prepared_ground_truth = ground_truth
                if len(prepared_ground_truth) > 300:
                    prepared_ground_truth = ground_truth[:300] + "..."
                
                # Query completo (sin truncar)
                prepared_query = user_query
                
                print(f"📊 Datos preparados:")
                print(f"   Query: {len(prepared_query)} chars")
                print(f"   Answer: {len(prepared_response)} chars")
                print(f"   Contexts: {len(prepared_contexts)} items")
                print(f"   Ground truth: {len(prepared_ground_truth)} chars")
                
                # ✅ CREAR DATASET RAGAS
                data = {
                    "question": [prepared_query],
                    "answer": [prepared_response],
                    "contexts": [prepared_contexts],
                    "ground_truth": [prepared_ground_truth]
                }
                
                # ✅ VALIDACIÓN COMPLETA DEL DATASET
                print(f"🔍 === VALIDANDO DATASET RAGAS ===")
                
                # Verificar estructura
                for key, value in data.items():
                    if not isinstance(value, list) or len(value) != 1:
                        print(f"❌ {key} debe ser una lista con 1 elemento")
                        raise ValueError(f"Dataset inválido: {key}")
                    print(f"✅ {key}: {type(value[0])} con contenido válido")
                
                # Verificar contenido de contexts
                if not isinstance(data["contexts"][0], list) or len(data["contexts"][0]) == 0:
                    print(f"❌ Contexts debe ser lista de strings")
                    raise ValueError("Contexts inválidos")
                
                print(f"✅ Dataset válido para RAGAS")
                
                # Crear dataset de HuggingFace
                dataset = Dataset.from_dict(data)
                print(f"✅ Dataset HuggingFace creado: {dataset}")
                
                # ✅ TEST PREVIO DE CONECTIVIDAD
                print(f"🔍 === TESTE DE CONECTIVIDAD ===")
                
                # Test LLM directo
                try:
                    test_response = judge_llm.complete("What is 2+2?")
                    print(f"✅ LLM directo funciona: {str(test_response)[:50]}...")
                except Exception as llm_error:
                    print(f"❌ LLM directo falla: {llm_error}")
                    continue
                
                # Test embeddings directo
                try:
                    test_embedding = embed_model.get_text_embedding("test")
                    print(f"✅ Embeddings directo funciona: {len(test_embedding)} dims")
                except Exception as embed_error:
                    print(f"❌ Embeddings directo falla: {embed_error}")
                    continue
                
                # ✅ EVALUACIÓN CON DEBUGGING DETALLADO
                print(f"🔄 === EVALUANDO MÉTRICAS RAGAS ===")
                
                individual_results = {}
                
                # Evaluar faithfulness
                print(f"   🔄 Evaluando faithfulness...")
                try:
                    # Verificar que el LLM está configurado
                    if not hasattr(faithfulness, 'llm') or faithfulness.llm is None:
                        print(f"   ❌ Faithfulness no tiene LLM configurado")
                        individual_results["faithfulness"] = 0.0
                    else:
                        print(f"   ✅ Faithfulness LLM: {type(faithfulness.llm)}")
                        
                        # Evaluar con timeout y debugging
                        faithfulness_result = evaluate(
                            dataset=dataset,
                            metrics=[faithfulness],
                            run_config=ragas_run_config
                        )
                        
                        raw_score = faithfulness_result["faithfulness"]
                        print(f"   ✅ Faithfulness raw: {raw_score} (tipo: {type(raw_score)})")
                        
                        # Procesar el resultado
                        if isinstance(raw_score, list):
                            processed_score = raw_score[0] if len(raw_score) > 0 else 0.0
                        else:
                            processed_score = float(raw_score) if raw_score is not None else 0.0
                            
                        # Verificar NaN
                        import math
                        if math.isnan(processed_score):
                            print(f"   ❌ Faithfulness devolvió NaN")
                            processed_score = 0.0
                        
                        individual_results["faithfulness"] = round(processed_score, 4)
                        print(f"   ✅ Faithfulness procesado: {individual_results['faithfulness']}")
                        
                except Exception as faith_error:
                    print(f"   ❌ Error en faithfulness: {faith_error}")
                    print(f"   Traceback: {traceback.format_exc()[:400]}...")
                    individual_results["faithfulness"] = 0.0
                
                # Evaluar context_recall
                print(f"   🔄 Evaluando context_recall...")
                try:
                    # Verificar configuración
                    if not hasattr(context_recall, 'llm') or context_recall.llm is None:
                        print(f"   ❌ Context recall no tiene LLM configurado")
                        individual_results["context_recall"] = 0.0
                    elif not hasattr(context_recall, 'embeddings') or context_recall.embeddings is None:
                        print(f"   ❌ Context recall no tiene embeddings configurado")
                        individual_results["context_recall"] = 0.0
                    else:
                        print(f"   ✅ Context recall LLM: {type(context_recall.llm)}")
                        print(f"   ✅ Context recall embeddings: {type(context_recall.embeddings)}")
                        
                        # Evaluar con timeout y debugging
                        recall_result = evaluate(
                            dataset=dataset,
                            metrics=[context_recall],
                            run_config=ragas_run_config
                        )
                        
                        raw_score = recall_result["context_recall"]
                        print(f"   ✅ Context recall raw: {raw_score} (tipo: {type(raw_score)})")
                        
                        # Procesar el resultado
                        if isinstance(raw_score, list):
                            processed_score = raw_score[0] if len(raw_score) > 0 else 0.0
                        else:
                            processed_score = float(raw_score) if raw_score is not None else 0.0
                            
                        # Verificar NaN
                        import math
                        if math.isnan(processed_score):
                            print(f"   ❌ Context recall devolvió NaN")
                            processed_score = 0.0
                        
                        individual_results["context_recall"] = round(processed_score, 4)
                        print(f"   ✅ Context recall procesado: {individual_results['context_recall']}")
                        
                except Exception as recall_error:
                    print(f"   ❌ Error en context_recall: {recall_error}")
                    print(f"   Traceback: {traceback.format_exc()[:400]}...")
                    individual_results["context_recall"] = 0.0
                
                # ✅ COMPILAR RESULTADOS FINALES
                print(f"🔍 === RESULTADOS FINALES PARA {model_name} ===")
                
                final_metrics = {}
                for metric_name, score in individual_results.items():
                    ragas_key = f"ragas_{metric_name}"
                    final_metrics[ragas_key] = score
                    print(f"   {ragas_key}: {score}")
                
                # Verificar que al menos una métrica funcionó
                valid_scores = [score for score in final_metrics.values() if score > 0.0]
                if len(valid_scores) == 0:
                    print(f"❌ Ninguna métrica RAGAS funcionó para {model_name}")
                    print(f"🔍 Datos del dataset:")
                    print(f"   Question: {data['question'][0][:100]}...")
                    print(f"   Answer: {data['answer'][0][:100]}...")
                    print(f"   Context[0]: {data['contexts'][0][0][:100]}...")
                    print(f"   Ground truth: {data['ground_truth'][0][:100]}...")
                else:
                    print(f"✅ {len(valid_scores)} métricas RAGAS válidas para {model_name}")
                
                ragas_results[model_name] = final_metrics
                
            except Exception as e:
                print(f"❌ Error general para {model_name}: {e}")
                print(f"Traceback completo: {traceback.format_exc()}")
                ragas_results[model_name] = {
                    "ragas_faithfulness": 0.0,
                    "ragas_context_recall": 0.0,
                    "ragas_error": str(e)
                }
        
        print(f"\n✅ === RESUMEN FINAL RAGAS ===")
        for model_name, results in ragas_results.items():
            print(f"📊 {model_name}: {results}")
        
        return ragas_results
        
    except Exception as e:
        print(f"❌ Error general en RAGAS: {e}")
        print(f"Traceback completo: {traceback.format_exc()}")
        return {}

def validate_ragas_inputs(user_query: str, contexts: List[str], answer: str) -> bool:
    """
    Valida que los inputs para RAGAS sean válidos.
    """
    if not user_query or not user_query.strip():
        return False
    if not answer or not answer.strip():
        return False
    if not contexts or len(contexts) == 0:
        return False
    return True