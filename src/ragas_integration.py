# En ragas_integration.py

import os
import math
import logging
from typing import Dict, List
from datasets import Dataset

# Imports de RAGAS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper

# ✅ CAMBIO: Usar el nuevo paquete langchain-ollama
try:
    from langchain_ollama import OllamaLLM
    print("✅ Usando langchain-ollama (recomendado)")
except ImportError:
    # Fallback al viejo si no está instalado
    from langchain_community.llms import Ollama as OllamaLLM
    print("⚠️ Usando langchain_community.llms.Ollama (deprecado)")

# Imports de tu proyecto
from ragas.embeddings.base import LlamaIndexEmbeddingsWrapper
from cache_manager import get_cache_manager

logger = logging.getLogger(__name__)

def calculate_ragas_metrics(user_query, model_responses, contexts, judge_response, config=None):
    """
    Calcula métricas RAGAS usando el wrapper de LangChain para Ollama,
    compatible con ragas==0.2.0 y con manejo correcto de timeouts.
    """
    try:
        os.environ["RAGAS_DO_NOT_TRACK"] = "true"
        
        judge_model_name = config.get("judge_model") if config else None
        llm_url = config.get("llm_url") if config else "http://localhost:11434"
        
        if not judge_model_name:
            print("❌ No se proporcionó modelo juez para RAGAS")
            return {}
        
        print(f"🎯 RAGAS usando modelo juez: {judge_model_name}")

        # ✅ CORRECCIÓN: Usar parámetros correctos según la versión
        try:
            # Intentar con el nuevo paquete langchain-ollama
            llm_langchain = OllamaLLM(
                model=judge_model_name,
                base_url=llm_url,
                timeout=1800,  # Nuevo parámetro para langchain-ollama
            )
            print("✅ Usando OllamaLLM (langchain-ollama)")
        except TypeError:
            # Fallback para langchain_community (parámetros diferentes)
            try:
                llm_langchain = OllamaLLM(
                    model=judge_model_name,
                    base_url=llm_url,
                    timeout=1800,  # Solo timeout para community
                )
                print("✅ Usando Ollama (langchain_community) - solo timeout")
            except TypeError:
                # Último fallback sin timeout personalizado
                llm_langchain = OllamaLLM(
                    model=judge_model_name,
                    base_url=llm_url,
                )
                print("⚠️ Usando Ollama sin timeout personalizado")
        
        # ✅ CREAR WRAPPER para RAGAS
        ragas_llm = LangchainLLMWrapper(llm=llm_langchain)
        
        # El modelo de embeddings sigue siendo el de LlamaIndex
        cache_manager = get_cache_manager()
        embed_model = cache_manager.get_cached_embedding_model()
        ragas_embeddings = LlamaIndexEmbeddingsWrapper(embeddings=embed_model)
        
        # Asignar los componentes a cada métrica
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings
        context_precision.llm = ragas_llm
        context_recall.llm = ragas_llm
        
        print(f"✅ RAGAS configurado con LangchainLLMWrapper para Ollama.")
        
        ragas_results = {}
        
        for model_name, response_text in model_responses.items():
            try:
                ground_truth = str(judge_response).strip() if judge_response else f"Authoritative answer for: {user_query}"
                if not response_text or not response_text.strip():
                     print(f"⚠️ Respuesta vacía para {model_name}, omitiendo RAGAS.")
                     continue
                
                dataset = Dataset.from_dict({
                    "question": [user_query],
                    "answer": [response_text],
                    "contexts": [contexts if contexts else ["No context provided"]],
                    "ground_truth": [ground_truth]
                })
                
                print(f"🔄 Evaluando RAGAS para el modelo: {model_name}...")
                
                result = evaluate(
                    dataset=dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                    raise_exceptions=False
                )
                
                print(f"✅ Evaluación RAGAS completada para {model_name}: {result}")
                
                processed_scores = {}
                for metric_name, score in result.items():
                    key = f"ragas_{metric_name}"
                    if isinstance(score, (float, int)) and not math.isnan(score):
                        processed_scores[key] = round(float(score), 4)
                    else:
                        print(f"⚠️ Métrica '{metric_name}' devolvió un valor inválido ({score}), se usará 0.0")
                        processed_scores[key] = 0.0
                ragas_results[model_name] = processed_scores
                
            except Exception as e:
                print(f"❌ Error evaluando RAGAS para {model_name}: {e}")
                ragas_results[model_name] = {"ragas_error": str(e)}
        
        return ragas_results
        
    except Exception as e:
        print(f"❌ Error general en la configuración de RAGAS: {e}")
        import traceback
        traceback.print_exc()
        return {}