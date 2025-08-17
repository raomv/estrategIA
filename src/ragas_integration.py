"""
Integración de métricas RAGAS para evaluación estándar.
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def calculate_ragas_metrics(user_query, model_responses, contexts, judge_response, config=None):
    """
    Calcula métricas RAGAS usando el MISMO modelo juez seleccionado por el usuario
    RAGAS 0.2.0 compatible
    """
    try:
        # ✅ USAR EL MISMO PATRÓN QUE LAS MÉTRICAS NATIVAS
        judge_model_name = config.get("judge_model") if config else None
        llm_url = config.get("llm_url") if config else "http://localhost:11434"
        
        if not judge_model_name:
            print("❌ No se proporcionó modelo juez para RAGAS")
            return {}
        
        print(f"🎯 RAGAS usando modelo juez seleccionado: {judge_model_name}")
        
        # ✅ IMPORTS CORRECTOS PARA RAGAS 0.2.0
        try:
            # Intentar diferentes rutas de import para RAGAS 0.2.0
            from ragas.llms.llamaindex import LlamaIndexLLM
            from ragas.embeddings.llamaindex import LlamaIndexEmbeddings
            print("✅ Imports directos funcionando")
        except ImportError:
            try:
                # Alternativa para RAGAS 0.2.0
                from ragas.llms import llamaindex_llm
                from ragas.embeddings import llamaindex_embeddings
                LlamaIndexLLM = llamaindex_llm.LlamaIndexLLM
                LlamaIndexEmbeddings = llamaindex_embeddings.LlamaIndexEmbeddings
                print("✅ Imports alternativos funcionando")
            except ImportError:
                # Si nada funciona, usar LangChain directamente
                print("⚠️ LlamaIndex wrappers no disponibles, usando LangChain")
                return calculate_ragas_with_langchain(user_query, model_responses, contexts, judge_response, config)
        
        # ✅ CREAR EL MISMO LLM QUE USAN LAS MÉTRICAS NATIVAS
        from llama_index.llms.ollama import Ollama
        judge_llm = Ollama(model=judge_model_name, base_url=llm_url, request_timeout=300.0)
        
        # ✅ CONFIGURAR RAGAS CON ESE LLM
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        
        ragas_llm = LlamaIndexLLM(llm=judge_llm)
        
        # ✅ USAR EL MISMO EMBED_MODEL QUE EL PROYECTO
        from cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        embed_model = cache_manager.get_cached_embedding_model()
        ragas_embeddings = LlamaIndexEmbeddings(embeddings=embed_model)
        
        # Configurar métricas RAGAS
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings
        context_precision.llm = ragas_llm
        context_recall.llm = ragas_llm
        context_recall.embeddings = ragas_embeddings
        
        print(f"✅ RAGAS configurado con juez: {judge_model_name}")
        
        # Calcular métricas
        ragas_results = {}
        
        for model_name, response_text in model_responses.items():
            try:
                from datasets import Dataset
                
                data = {
                    "question": [user_query],
                    "answer": [response_text],
                    "contexts": [contexts if contexts else ["No context available"]],
                    "ground_truth": [judge_response]
                }
                
                dataset = Dataset.from_dict(data)
                
                result = evaluate(
                    dataset=dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
                )
                
                ragas_results[model_name] = {
                    "ragas_faithfulness": float(result["faithfulness"]),
                    "ragas_answer_relevancy": float(result["answer_relevancy"]),
                    "ragas_context_precision": float(result["context_precision"]),
                    "ragas_context_recall": float(result["context_recall"])
                }
                
                print(f"✅ RAGAS calculado para {model_name}")
                
            except Exception as e:
                print(f"❌ Error calculating RAGAS metrics for {model_name}: {e}")
                ragas_results[model_name] = {"ragas_error": str(e)}
        
        return ragas_results
        
    except Exception as e:
        print(f"❌ Error general en RAGAS: {e}")
        import traceback
        traceback.print_exc()
        return {}

def calculate_ragas_with_langchain(user_query, model_responses, contexts, judge_response, config):
    """
    Fallback usando LangChain directamente para RAGAS 0.2.0
    """
    try:
        print("🔄 Usando fallback con LangChain para RAGAS...")
        
        # Usar LangChain Ollama directamente
        from langchain_community.llms import Ollama as LangChainOllama
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        
        judge_model_name = config.get("judge_model")
        llm_url = config.get("llm_url", "http://localhost:11434")
        
        # Configurar LangChain LLM
        langchain_llm = LangChainOllama(
            model=judge_model_name,
            base_url=llm_url,
            timeout=300
        )
        
        # Configurar métricas RAGAS con LangChain LLM
        faithfulness.llm = langchain_llm
        answer_relevancy.llm = langchain_llm
        context_precision.llm = langchain_llm
        context_recall.llm = langchain_llm
        
        print(f"✅ RAGAS configurado con LangChain: {judge_model_name}")
        
        # Resto igual...
        ragas_results = {}
        
        for model_name, response_text in model_responses.items():
            try:
                from datasets import Dataset
                
                data = {
                    "question": [user_query],
                    "answer": [response_text],
                    "contexts": [contexts if contexts else ["No context available"]],
                    "ground_truth": [judge_response]
                }
                
                dataset = Dataset.from_dict(data)
                
                result = evaluate(
                    dataset=dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
                )
                
                ragas_results[model_name] = {
                    "ragas_faithfulness": float(result["faithfulness"]),
                    "ragas_answer_relevancy": float(result["answer_relevancy"]),
                    "ragas_context_precision": float(result["context_precision"]),
                    "ragas_context_recall": float(result["context_recall"])
                }
                
                print(f"✅ RAGAS calculado para {model_name}")
                
            except Exception as e:
                print(f"❌ Error calculating RAGAS metrics for {model_name}: {e}")
                ragas_results[model_name] = {"ragas_error": str(e)}
        
        return ragas_results
        
    except Exception as e:
        print(f"❌ Error en fallback LangChain: {e}")
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