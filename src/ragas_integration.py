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
        judge_llm = Ollama(model=judge_model_name, base_url=llm_url, request_timeout=300.0)
        
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
                
                # 🔍 DEBUG COMPLETO - VER FORMATO EXACTO DE RAGAS
                print(f"\n🔍 === DEBUG RAGAS PARA {model_name} ===")
                print(f"Tipo de resultado: {type(result)}")
                print(f"Resultado completo: {result}")
                
                if hasattr(result, 'keys'):
                    print(f"Keys disponibles: {list(result.keys())}")
                    
                    for key in result.keys():
                        value = result[key]
                        print(f"  {key}:")
                        print(f"    Valor: {value}")
                        print(f"    Tipo: {type(value)}")
                        if isinstance(value, list):
                            print(f"    Longitud lista: {len(value)}")
                            if len(value) > 0:
                                print(f"    Primer elemento: {value[0]} (tipo: {type(value[0])})")
                        elif hasattr(value, '__iter__') and not isinstance(value, str):
                            print(f"    Es iterable: Sí")
                            try:
                                list_values = list(value)
                                print(f"    Como lista: {list_values}")
                            except:
                                print(f"    No se pudo convertir a lista")
                
                print(f"🔍 === FIN DEBUG PARA {model_name} ===\n")
                
                # TEMPORALMENTE - Solo devolver error para ver el debug
                ragas_results[model_name] = {"debug_completed": True, "raw_result": str(result)}
                
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