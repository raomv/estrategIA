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
                
                # ✅ DEBUG COMPLETO - Ver exactamente qué recibe RAGAS
                print(f"\n🔍 === DEBUG COMPLETO PARA {model_name} ===")
                print(f"📝 User query ({len(user_query)} chars): {user_query}")
                print(f"📝 Model response ({len(response_text)} chars): {response_text[:200]}...")
                print(f"📝 Judge response ({len(str(judge_response)) if judge_response else 0} chars): {str(judge_response)[:200] if judge_response else 'NONE'}...")
                print(f"📝 Contexts count: {len(contexts)}")
                
                if contexts:
                    for i, ctx in enumerate(contexts[:3]):  # Mostrar primeros 3 contextos
                        print(f"   📄 Context {i+1} ({len(ctx)} chars): {ctx[:150]}...")
                else:
                    print("   ❌ NO HAY CONTEXTOS")
                
                # ✅ VERIFICAR CALIDAD DE DATOS
                if not user_query or len(user_query.strip()) < 5:
                    print(f"❌ PROBLEMA: User query muy corto o vacío")
                
                if not response_text or len(response_text.strip()) < 10:
                    print(f"❌ PROBLEMA: Response muy corta o vacía")
                
                if not judge_response or len(str(judge_response).strip()) < 10:
                    print(f"❌ PROBLEMA: Judge response muy corto o vacío: '{judge_response}'")
                
                if not contexts or len(contexts) == 0:
                    print(f"❌ PROBLEMA: No hay contextos")
                
                # ✅ CREAR DATASET CON VERIFICACIÓN
                data = {
                    "question": [user_query],
                    "answer": [response_text],
                    "contexts": [contexts if contexts else ["No context available"]],
                    "ground_truth": [str(judge_response)] if judge_response else [response_text]  # Fallback
                }
                
                print(f"📊 Dataset creado:")
                print(f"   question: '{data['question'][0]}'")
                print(f"   answer: '{data['answer'][0][:100]}...'")
                print(f"   contexts: {len(data['contexts'][0])} items")
                print(f"   ground_truth: '{data['ground_truth'][0][:100]}...'")
                
                dataset = Dataset.from_dict(data)
                print(f"✅ Dataset HuggingFace creado correctamente")
                
                # ✅ EVALUAR CON DEBUG DETALLADO
                print(f"🔄 Iniciando evaluación RAGAS...")
                
                try:
                    result = evaluate(
                        dataset=dataset,
                        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
                    )
                    print(f"✅ Evaluación RAGAS completada")
                    
                    # ✅ DEBUG: Inspeccionar resultado crudo
                    print(f"🔍 Resultado crudo RAGAS:")
                    print(f"   Tipo: {type(result)}")
                    print(f"   Contenido: {result}")
                    
                    if hasattr(result, 'keys'):
                        print(f"   Keys disponibles: {list(result.keys())}")
                        for key in result.keys():
                            value = result[key]
                            print(f"   {key}: {value} (tipo: {type(value)})")
                    
                except Exception as eval_error:
                    print(f"❌ ERROR EN EVALUACIÓN RAGAS: {eval_error}")
                    import traceback
                    traceback.print_exc()
                    
                    # Crear resultado por defecto para continuar debug
                    result = {
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                        "context_precision": 0.0,
                        "context_recall": 0.0
                    }
                
                # ✅ SANITIZACIÓN CON DEBUG
                def sanitize_ragas_value(value):
                    print(f"      🔧 Sanitizando: {value} (tipo: {type(value)})")
                    
                    import math
                    import numpy as np
                    
                    original_value = value
                    
                    # Si es lista, tomar primer elemento
                    if isinstance(value, list):
                        if len(value) > 0:
                            value = value[0]
                            print(f"         Lista → primer elemento: {value}")
                        else:
                            print(f"         Lista vacía → 0.0")
                            return 0.0
                    
                    # Convertir numpy types a Python natives
                    if isinstance(value, np.ndarray):
                        value = float(value.item())
                        print(f"         ndarray → float: {value}")
                    elif hasattr(value, 'item'):  # numpy scalar
                        value = float(value.item())
                        print(f"         numpy scalar → float: {value}")
                    elif isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
                        value = float(value)
                        print(f"         numpy type → float: {value}")
                    
                    # Manejar NaN, inf, -inf
                    if isinstance(value, (int, float)):
                        if math.isnan(value):
                            print(f"         NaN detectado → 0.0")
                            return 0.0
                        elif math.isinf(value):
                            print(f"         Inf detectado → 0.0")
                            return 0.0
                        else:
                            sanitized = round(float(value), 4)
                            print(f"         Valor válido: {sanitized}")
                            return sanitized
                    
                    # Fallback
                    print(f"         Tipo no reconocido: {type(value)} → 0.0")
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