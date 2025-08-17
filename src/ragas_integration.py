"""
Integración de métricas RAGAS para evaluación estándar.
"""
import logging
from typing import Dict, List, Any, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

logger = logging.getLogger(__name__)

def calculate_ragas_metrics(
    user_query: str, 
    model_responses: Dict[str, Dict], 
    contexts: List[str], 
    judge_response: str
) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas RAGAS usando datos ya disponibles del flujo actual.
    
    Args:
        user_query: Pregunta del usuario
        model_responses: Respuestas de todos los modelos evaluados
        contexts: Documentos recuperados por el retriever
        judge_response: Respuesta del modelo juez usada como ground truth
        
    Returns:
        Dict con métricas RAGAS para cada modelo
    """
    try:
        ragas_results = {}
        
        # Preparar contextos como lista de strings
        context_list = []
        if isinstance(contexts, list):
            context_list = [str(ctx) for ctx in contexts]
        else:
            context_list = [str(contexts)] if contexts else [""]
        
        for model_name, response_data in model_responses.items():
            try:
                # Extraer respuesta del modelo
                model_answer = ""
                if isinstance(response_data, dict) and 'response' in response_data:
                    if hasattr(response_data['response'], 'response'):
                        model_answer = str(response_data['response'].response)
                    else:
                        model_answer = str(response_data['response'])
                else:
                    model_answer = str(response_data)
                
                # Preparar dataset para RAGAS
                data = {
                    "question": [user_query],
                    "answer": [model_answer],
                    "contexts": [context_list],
                    "ground_truth": [str(judge_response)]
                }
                
                # Crear dataset
                dataset = Dataset.from_dict(data)
                
                # Evaluar con RAGAS
                result = evaluate(
                    dataset=dataset,
                    metrics=[context_precision, context_recall]
                )
                
                # Extraer resultados
                ragas_results[model_name] = {
                    "ragas_context_precision": float(result["context_precision"][0]) if result["context_precision"] else 0.0,
                    "ragas_context_recall": float(result["context_recall"][0]) if result["context_recall"] else 0.0
                }
                
                logger.info(f"RAGAS metrics calculated for {model_name}: {ragas_results[model_name]}")
                
            except Exception as e:
                logger.error(f"Error calculating RAGAS metrics for {model_name}: {e}")
                ragas_results[model_name] = {
                    "ragas_context_precision": 0.0,
                    "ragas_context_recall": 0.0
                }
        
        return ragas_results
        
    except Exception as e:
        logger.error(f"Error in RAGAS evaluation: {e}")
        return {}

def validate_ragas_inputs(user_query: str, contexts: List[str], answer: str) -> bool:
    """
    Valida que los inputs para RAGAS sean válidos.
    
    Returns:
        True si los inputs son válidos
    """
    if not user_query or not user_query.strip():
        return False
    if not answer or not answer.strip():
        return False
    if not contexts or len(contexts) == 0:
        return False
    return True