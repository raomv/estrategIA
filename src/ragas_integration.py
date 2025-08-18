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
                
                data = {
                    "question": [user_query],
                    "answer": [response_text],
                    "contexts": [contexts if contexts else ["No context available"]],
                    "ground_truth": [judge_response] if judge_response else [""]
                }
                
                dataset = Dataset.from_dict(data)
                
                result = evaluate(
                    dataset=dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
                )
                
                # ✅ SANITIZAR RESULTADOS PARA JSON
                def sanitize_ragas_value(value):
                    """Convierte valores RAGAS a formato JSON-serializable"""
                    import math
                    import numpy as np
                    
                    # Si es lista, tomar primer elemento
                    if isinstance(value, list):
                        if len(value) > 0:
                            value = value[0]
                        else:
                            return 0.0
                    
                    # Convertir numpy types a Python natives
                    if isinstance(value, np.ndarray):
                        value = float(value.item())
                    elif hasattr(value, 'item'):  # numpy scalar
                        value = float(value.item())
                    elif isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
                        value = float(value)
                    
                    # Manejar NaN, inf, -inf
                    if isinstance(value, (int, float)):
                        if math.isnan(value) or math.isinf(value):
                            return 0.0
                        return round(float(value), 4)
                    
                    # Fallback
                    return 0.0
                
                # ✅ PROCESAR RESULTADOS SANITIZADOS - ACCESO DIRECTO
                try:
                    ragas_results[model_name] = {
                        "ragas_faithfulness": sanitize_ragas_value(result["faithfulness"]),
                        "ragas_answer_relevancy": sanitize_ragas_value(result["answer_relevancy"]),
                        "ragas_context_precision": sanitize_ragas_value(result["context_precision"]),
                        "ragas_context_recall": sanitize_ragas_value(result["context_recall"])
                    }
                except KeyError as e:
                    print(f"⚠️ Métrica faltante en resultado RAGAS: {e}")
                    ragas_results[model_name] = {
                        "ragas_faithfulness": 0.0,
                        "ragas_answer_relevancy": 0.0,
                        "ragas_context_precision": 0.0,
                        "ragas_context_recall": 0.0,
                        "ragas_error": f"Missing metric: {e}"
                    }
                
                print(f"✅ RAGAS calculado para {model_name}: {ragas_results[model_name]}")
                
            except Exception as e:
                print(f"❌ Error calculating RAGAS metrics for {model_name}: {e}")
                ragas_results[model_name] = {"ragas_error": str(e)}
        
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