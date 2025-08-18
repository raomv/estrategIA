"""
Integración de métricas RAGAS para evaluación estándar.
RAGAS 0.2.0 con LlamaIndex (sin LangChain)
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def calculate_ragas_metrics(user_query, model_responses, contexts, judge_response, config=None):
    """
    Calcula métricas RAGAS usando el MISMO modelo juez seleccionado por el usuario
    RAGAS 0.2.0 + LlamaIndex (sin LangChain)
    """
    try:
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
        
        # ✅ CREAR EL MISMO LLM QUE USAN LAS MÉTRICAS NATIVAS
        from llama_index.llms.ollama import Ollama
        judge_llm = Ollama(model=judge_model_name, base_url=llm_url, request_timeout=600.0)
        
        # ✅ CONFIGURAR RAGAS CON ESE LLM (SOLO LLAMAINDEX)
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        
        # Usar los wrappers correctos
        ragas_llm = LlamaIndexLLMWrapper(llm=judge_llm)
        
        # ✅ USAR EL MISMO EMBED_MODEL QUE EL PROYECTO
        from cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        embed_model = cache_manager.get_cached_embedding_model()
        ragas_embeddings = LlamaIndexEmbeddingsWrapper(embeddings=embed_model)
        
        # Configurar métricas RAGAS
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings
        context_precision.llm = ragas_llm
        context_recall.llm = ragas_llm
        context_recall.embeddings = ragas_embeddings
        
        print(f"✅ RAGAS configurado con juez LlamaIndex: {judge_model_name}")
        
        # Calcular métricas
        ragas_results = {}
        
        for model_name, response_text in model_responses.items():
            try:
                from datasets import Dataset
                
                # ✅ DEBUG MEJORADO PARA VERIFICAR RESPUESTA DE REFERENCIA
                print(f"\n🔍 === DEBUG COMPLETO PARA {model_name} ===")
                print(f"📝 User query ({len(user_query)} chars): {user_query}")
                print(f"📝 Model response ({len(response_text)} chars): {response_text[:200]}...")
                
                # ✅ DEBUG ESPECÍFICO DE JUDGE RESPONSE
                if judge_response:
                    print(f"📝 Judge reference ({len(str(judge_response))} chars): {str(judge_response)[:200]}...")
                    print(f"✅ Judge response válido: {len(str(judge_response).strip()) > 20}")
                else:
                    print(f"❌ Judge response: NONE - Se usará ground truth alternativo")
                
                print(f"📝 Contexts count: {len(contexts)}")
                
                # ✅ GROUND TRUTH CON PREFERENCIA POR RESPUESTA DEL JUEZ
                if judge_response and len(str(judge_response).strip()) > 20:
                    ground_truth = str(judge_response).strip()
                    print(f"✅ Usando respuesta del juez como ground truth")
                    print(f"📝 Ground truth (juez): {ground_truth[:150]}...")
                else:
                    ground_truth = f"A comprehensive answer addressing: {user_query}"
                    print(f"⚠️ Usando ground truth alternativo")
                    print(f"📝 Ground truth (alternativo): {ground_truth}")
                
                # ✅ CREAR DATASET CON VERIFICACIÓN
                data = {
                    "question": [user_query],
                    "answer": [response_text],
                    "contexts": [contexts if contexts else ["No context available"]],
                    "ground_truth": [ground_truth]  # ← ÚNICO CAMBIO AQUÍ
                }
                
                print(f"📊 Dataset creado:")
                print(f"   question: '{data['question'][0]}'")
                print(f"   answer: '{data['answer'][0][:100]}...'")
                print(f"   contexts: {len(data['contexts'][0])} items")
                print(f"   ground_truth: '{data['ground_truth'][0][:100]}...'")
                
                dataset = Dataset.from_dict(data)
                print(f"✅ Dataset HuggingFace creado correctamente")
                
                # ✅ DEBUG PREVIO A EVALUACIÓN - Verificar configuración COMPLETA
                print(f"🔍 === VERIFICANDO CONFIGURACIÓN RAGAS ===")
                print(f"   Judge LLM configurado: {judge_llm}")
                print(f"   Embed model configurado: {embed_model}")
                print(f"   Faithfulness LLM: {getattr(faithfulness, 'llm', 'NO CONFIGURADO')}")
                print(f"   Answer relevancy LLM: {getattr(answer_relevancy, 'llm', 'NO CONFIGURADO')}")
                print(f"   Answer relevancy embeddings: {getattr(answer_relevancy, 'embeddings', 'NO CONFIGURADO')}")
                print(f"   Context precision LLM: {getattr(context_precision, 'llm', 'NO CONFIGURADO')}")
                print(f"   Context recall LLM: {getattr(context_recall, 'llm', 'NO CONFIGURADO')}")
                print(f"   Context recall embeddings: {getattr(context_recall, 'embeddings', 'NO CONFIGURADO')}")
                
                # ✅ TESTE DE COMPONENTES ANTES DE EVALUACIÓN
                print(f"🔍 === TESTE DE COMPONENTES ===")
                
                # Test modelo juez
                try:
                    test_response = judge_llm.complete("Test simple")
                    print(f"   ✅ Modelo juez responde: {str(test_response)[:50]}...")
                except Exception as test_error:
                    print(f"   ❌ Modelo juez no responde: {test_error}")
                
                # Test embeddings
                try:
                    test_embedding = embed_model.get_text_embedding("test embedding")
                    print(f"   ✅ Embeddings funcionan: {len(test_embedding)} dimensiones")
                except Exception as embed_error:
                    print(f"   ❌ Embeddings no funcionan: {embed_error}")
                
                # Test ragas wrappers
                try:
                    ragas_test = ragas_llm.complete("Test RAGAS wrapper")
                    print(f"   ✅ RAGAS LLM wrapper funciona: {str(ragas_test)[:50]}...")
                except Exception as wrapper_error:
                    print(f"   ❌ RAGAS LLM wrapper falla: {wrapper_error}")
                
                # ✅ VERIFICAR DATOS DEL DATASET MÁS DETALLADAMENTE
                print(f"🔍 === VERIFICANDO DATOS DATASET ===")
                print(f"   Question: '{data['question'][0]}'")
                print(f"   Question válida: {len(data['question'][0]) > 5}")
                print(f"   Answer length: {len(data['answer'][0])}")
                print(f"   Answer válida: {len(data['answer'][0]) > 10}")
                print(f"   Contexts count: {len(data['contexts'][0])}")
                print(f"   Contexts válidos: {len(data['contexts'][0]) > 0 and data['contexts'][0] != ['No context available']}")
                print(f"   Ground truth length: {len(data['ground_truth'][0])}")
                print(f"   Ground truth válido: {len(data['ground_truth'][0]) > 10}")
                
                # Mostrar contenido real
                if data['contexts'][0]:
                    print(f"   Primera context preview: {data['contexts'][0][0][:100]}...")
                print(f"   Ground truth preview: {data['ground_truth'][0][:100]}...")
                
                # ✅ EVALUAR UNA MÉTRICA A LA VEZ PARA IDENTIFICAR PROBLEMAS
                print(f"🔄 === EVALUACIÓN INDIVIDUAL DE MÉTRICAS ===")
                
                try:
                    individual_results = {}
                    
                    metrics_to_test = [
                        ("faithfulness", faithfulness),
                        ("answer_relevancy", answer_relevancy), 
                        ("context_precision", context_precision),
                        ("context_recall", context_recall)
                    ]
                    
                    for metric_name, metric_obj in metrics_to_test:
                        try:
                            print(f"   🔄 Evaluando {metric_name} individualmente...")
                            
                            # Verificar configuración específica de la métrica
                            if hasattr(metric_obj, 'llm'):
                                print(f"      LLM configurado: {metric_obj.llm is not None}")
                            if hasattr(metric_obj, 'embeddings'):
                                print(f"      Embeddings configurado: {metric_obj.embeddings is not None}")
                            
                            individual_result = evaluate(
                                dataset=dataset,
                                metrics=[metric_obj]
                            )
                            
                            value = individual_result[metric_name]
                            print(f"   ✅ {metric_name}: {value} (tipo: {type(value)})")
                            
                            # Verificar si es NaN
                            import math
                            if isinstance(value, float) and math.isnan(value):
                                print(f"      ⚠️ {metric_name} devolvió NaN - problema en configuración o datos")
                            
                            individual_results[metric_name] = value
                            
                        except Exception as individual_error:
                            print(f"   ❌ {metric_name} falló individualmente: {individual_error}")
                            import traceback
                            print(f"      Traceback: {traceback.format_exc()[:300]}...")
                            individual_results[metric_name] = float('nan')
                    
                    # Usar resultados individuales
                    result = individual_results
                    print(f"✅ Evaluación individual completada: {result}")
                    
                except Exception as eval_error:
                    print(f"❌ ERROR EN EVALUACIÓN INDIVIDUAL: {eval_error}")
                    import traceback
                    traceback.print_exc()
                    
                    # Crear resultado por defecto
                    result = {
                        "faithfulness": float('nan'),
                        "answer_relevancy": float('nan'),
                        "context_precision": float('nan'),
                        "context_recall": float('nan')
                    }
                
                # ✅ DEBUG: Inspeccionar resultado crudo
                print(f"🔍 Resultado crudo RAGAS:")
                print(f"   Tipo: {type(result)}")
                print(f"   Contenido: {result}")
                
                if hasattr(result, 'keys'):
                    print(f"   Keys disponibles: {list(result.keys())}")
                    for key in result.keys():
                        value = result[key]
                        print(f"   {key}: {value} (tipo: {type(value)})")
                    
                # ✅ SANITIZACIÓN MEJORADA PARA NaN
                def sanitize_ragas_value(value):
                    print(f"      🔧 Sanitizando: {value} (tipo: {type(value)})")
                    
                    import math
                    import numpy as np
                    
                    # ✅ MANEJAR NaN ESPECÍFICAMENTE PRIMERO
                    if isinstance(value, float) and math.isnan(value):
                        print(f"         ❌ NaN detectado - RAGAS no pudo calcular la métrica")
                        print(f"         Causas posibles: datos insuficientes, modelo juez no responde, o configuración incorrecta")
                        return 0.0
                    
                    # Si es lista, tomar primer elemento
                    if isinstance(value, list):
                        if len(value) > 0:
                            value = value[0]
                            print(f"         Lista → primer elemento: {value}")
                            # Verificar NaN en lista
                            if isinstance(value, float) and math.isnan(value):
                                print(f"         ❌ NaN en lista")
                                return 0.0
                        else:
                            print(f"         Lista vacía → 0.0")
                            return 0.0
                    
                    # Convertir numpy types
                    if isinstance(value, np.ndarray):
                        value = float(value.item())
                        print(f"         ndarray → float: {value}")
                    elif hasattr(value, 'item'):
                        value = float(value.item())
                        print(f"         numpy scalar → float: {value}")
                    elif isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
                        value = float(value)
                        print(f"         numpy type → float: {value}")
                    
                    # Verificar NaN DESPUÉS de conversiones
                    if isinstance(value, (int, float)):
                        if math.isnan(value):
                            print(f"         ❌ NaN después de conversión")
                            return 0.0
                        elif math.isinf(value):
                            print(f"         ❌ Inf detectado → 0.0")
                            return 0.0
                        else:
                            sanitized = round(float(value), 4)
                            print(f"         ✅ Valor válido: {sanitized}")
                            return sanitized
                    
                    print(f"         ❌ Tipo no reconocido: {type(value)} → 0.0")
                    return 0.0
                
                # ✅ PROCESAR CADA MÉTRICA INDIVIDUALMENTE
                print(f"🔧 Procesando métricas individuales:")
                
                ragas_results[model_name] = {}
                
                metrics_to_process = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                
                for metric_name in metrics_to_process:
                    try:
                        if metric_name in result:
                            raw_value = result[metric_name]
                            print(f"   📊 {metric_name}: {raw_value} (tipo: {type(raw_value)})")
                            sanitized_value = sanitize_ragas_value(raw_value)
                            ragas_results[model_name][f"ragas_{metric_name}"] = sanitized_value
                            print(f"      ✅ Sanitizado: {sanitized_value}")
                        else:
                            print(f"   ❌ {metric_name} no encontrado en resultado")
                            ragas_results[model_name][f"ragas_{metric_name}"] = 0.0
                    except Exception as metric_error:
                        print(f"   ❌ Error procesando {metric_name}: {metric_error}")
                        ragas_results[model_name][f"ragas_{metric_name}"] = 0.0
                
                print(f"✅ RAGAS procesado para {model_name}: {ragas_results[model_name]}")
                
            except Exception as e:
                print(f"❌ Error general para {model_name}: {e}")
                import traceback
                traceback.print_exc()
                ragas_results[model_name] = {
                    "ragas_faithfulness": 0.0,
                    "ragas_answer_relevancy": 0.0,
                    "ragas_context_precision": 0.0,
                    "ragas_context_recall": 0.0,
                    "ragas_error": str(e)
                }
        
        return ragas_results
        
    except Exception as e:
        print(f"❌ Error general en RAGAS: {e}")
        import traceback
        traceback.print_exc()
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