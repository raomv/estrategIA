"""
Integración de métricas RAGAS para evaluación estándar.
RAGAS 0.2.0 con LlamaIndex (sin LangChain)
"""
import logging
import os
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
            # ✅ APLICAR RUNCONFIG A LA MÉTRICA
            if hasattr(faithfulness, 'run_config'):
                faithfulness.run_config = ragas_run_config
                print(f"✅ Faithfulness configurado con RunConfig extendido")
            else:
                print(f"⚠️ Faithfulness no tiene atributo run_config")
        except Exception as e:
            print(f"❌ Error configurando faithfulness: {e}")

        # Para context_recall: necesita question, ground_truth, contexts, embeddings
        try:
            context_recall.llm = ragas_llm
            context_recall.embeddings = ragas_embeddings
            # ✅ APLICAR RUNCONFIG A LA MÉTRICA
            if hasattr(context_recall, 'run_config'):
                context_recall.run_config = ragas_run_config
                print(f"✅ Context recall configurado con RunConfig extendido")
            else:
                print(f"⚠️ Context recall no tiene atributo run_config")
        except Exception as e:
            print(f"❌ Error configurando context_recall: {e}")

        print(f"✅ RAGAS configurado con juez: {judge_model_name} y RunConfig extendido")
        
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
                # ✅ VERIFICAR QUE GROUND TRUTH SEA DIFERENTE DEL ANSWER
                if judge_response and len(str(judge_response).strip()) > 20:
                    judge_text = str(judge_response).strip()
                    
                    # ✅ VERIFICAR QUE NO SEAN IDÉNTICOS
                    if judge_text == response_text:
                        print(f"⚠️ Judge response idéntico al model response - usando ground truth alternativo")
                        ground_truth = f"A comprehensive, authoritative answer addressing: {user_query}"
                    else:
                        ground_truth = judge_text
                        print(f"✅ Usando respuesta del juez como ground truth (diferente del modelo)")
                        print(f"📝 Ground truth (juez): {ground_truth[:150]}...")
                else:
                    ground_truth = f"A comprehensive answer addressing: {user_query}"
                    print(f"⚠️ Usando ground truth alternativo")
                    print(f"📝 Ground truth (alternativo): {ground_truth}")
                
                # ✅ OPTIMIZAR DATOS PARA RAGAS - REDUCIR COMPLEJIDAD
                print(f"🔄 === OPTIMIZANDO DATOS PARA RAGAS (timeout 180s) ===")

                # ✅ LIMITAR CONTEXTOS A MÁXIMO 2 FRAGMENTOS MÁS CORTOS
                limited_contexts = contexts[:2] if contexts else ["No context available"]
                truncated_contexts = []

                for ctx in limited_contexts:
                    # ✅ TRUNCAR CADA CONTEXTO A MÁXIMO 300 CARACTERES
                    if len(ctx) > 300:
                        truncated_ctx = ctx[:300] + "..."
                        truncated_contexts.append(truncated_ctx)
                        print(f"   ✂️ Contexto truncado: {len(ctx)} → {len(truncated_ctx)} chars")
                    else:
                        truncated_contexts.append(ctx)
                        print(f"   ✅ Contexto mantenido: {len(ctx)} chars")

                # ✅ TRUNCAR RESPUESTA DEL MODELO A MÁXIMO 200 CARACTERES
                if len(response_text) > 200:
                    truncated_response = response_text[:200] + "..."
                    print(f"   ✂️ Respuesta truncada: {len(response_text)} → {len(truncated_response)} chars")
                else:
                    truncated_response = response_text
                    print(f"   ✅ Respuesta mantenida: {len(response_text)} chars")

                # ✅ TRUNCAR GROUND TRUTH A MÁXIMO 150 CARACTERES
                if judge_response and len(str(judge_response).strip()) > 20:
                    judge_text = str(judge_response).strip()
                    if len(judge_text) > 150:
                        ground_truth = judge_text[:150] + "..."
                        print(f"   ✂️ Ground truth truncado: {len(judge_text)} → {len(ground_truth)} chars")
                    else:
                        ground_truth = judge_text
                        print(f"   ✅ Ground truth del juez mantenido: {len(ground_truth)} chars")
                else:
                    ground_truth = f"Answer to: {user_query[:50]}..."
                    print(f"   ⚠️ Ground truth alternativo corto: {len(ground_truth)} chars")

                # ✅ CREAR DATASET OPTIMIZADO
                data = {
                    "question": [user_query[:100]],  # ✅ TRUNCAR PREGUNTA TAMBIÉN
                    "answer": [truncated_response],
                    "contexts": [truncated_contexts],  # ✅ CONTEXTOS LIMITADOS Y TRUNCADOS
                    "ground_truth": [ground_truth]
                }

                print(f"📊 Dataset RAGAS optimizado:")
                print(f"   question: {len(data['question'][0])} chars")
                print(f"   answer: {len(data['answer'][0])} chars") 
                print(f"   contexts: {len(data['contexts'][0])} items, total: {sum(len(c) for c in data['contexts'][0])} chars")
                print(f"   ground_truth: {len(data['ground_truth'][0])} chars")
                print(f"   🎯 Total chars: {sum(len(str(v[0])) for v in data.values())} (objetivo: <800)")
                
                dataset = Dataset.from_dict(data)
                print(f"✅ Dataset HuggingFace creado correctamente")
                
                # ✅ DEBUG PREVIO A EVALUACIÓN - Verificar configuración COMPLETA
                print(f"🔍 === VERIFICANDO CONFIGURACIÓN RAGAS ===")
                print(f"   Judge LLM configurado: {judge_llm}")
                print(f"   Embed model configurado: {embed_model}")
                print(f"   Faithfulness LLM: {getattr(faithfulness, 'llm', 'NO CONFIGURADO')}")
                print(f"   Context recall LLM: {getattr(context_recall, 'llm', 'NO CONFIGURADO')}")
                print(f"   Context recall embeddings: {getattr(context_recall, 'embeddings', 'NO CONFIGURADO')}")
                # ✅ MÉTRICAS DESACTIVADAS:
                print(f"   ⚠️ Answer relevancy: DESACTIVADA (problemas de NaN)")
                print(f"   ⚠️ Context precision: DESACTIVADA (problemas de NaN)")
                
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
                    # ✅ USAR EL MÉTODO CORRECTO DE RAGAS 0.2.0
                    test_prompt = "Test RAGAS wrapper"
                    
                    # El wrapper de RAGAS usa 'generate' en lugar de 'complete'
                    if hasattr(ragas_llm, 'generate'):
                        ragas_test = ragas_llm.generate(test_prompt)
                        print(f"   ✅ RAGAS LLM wrapper funciona: {str(ragas_test)[:50]}...")
                    elif hasattr(ragas_llm, 'complete'):
                        ragas_test = ragas_llm.complete(test_prompt)
                        print(f"   ✅ RAGAS LLM wrapper funciona: {str(ragas_test)[:50]}...")
                    else:
                        # Solo verificar que el wrapper existe
                        print(f"   ✅ RAGAS LLM wrapper creado correctamente: {type(ragas_llm)}")
                        print(f"   📋 Métodos disponibles: {[m for m in dir(ragas_llm) if not m.startswith('_')]}")
                        
                except Exception as wrapper_error:
                    print(f"   ⚠️ RAGAS LLM wrapper test falló: {wrapper_error}")
                    print(f"   ℹ️ Esto es normal - el wrapper funciona para evaluate() pero no para test directo")
                
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
                
                # ✅ EVALUAR UNA MÉTRICA A LA VEZ PARA IDENTIFICAR PROBLEMAS CON TIMEOUT CONFIGURADO
                print(f"🔄 === EVALUACIÓN INDIVIDUAL DE MÉTRICAS CON TIMEOUT EXTENDIDO ===")
                
                try:
                    individual_results = {}
                    
                    metrics_to_test = [
                        ("faithfulness", faithfulness),
                        #("answer_relevancy", answer_relevancy), 
                        #("context_precision", context_precision),
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
                            
                            # ✅ CONFIGURACIÓN DE TIMEOUT EN EVALUATE
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
                
                metrics_to_process = ["faithfulness", "context_recall"]  # ✅ SOLO ESTAS DOS
                
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