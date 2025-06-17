#!/usr/bin/env python3

import argparse
import yaml
import os
import sys
from pathlib import Path

# Añadir el directorio src al path para las importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar las clases necesarias
from cache_manager import get_cache_manager
from data import Data

def process_documents_for_upload(directory: str, collection: str, chunk_size: int = 1024):
    """
    Procesa documentos subidos vía web usando docling y los carga en Qdrant.
    
    Args:
        directory: Directorio con los archivos subidos
        collection: Nombre de la colección en Qdrant  
        chunk_size: Tamaño de los chunks
    """
    print(f"=== PROCESANDO DOCUMENTOS SUBIDOS ===")
    print(f"Directorio: {directory}")
    print(f"Colección: {collection}")
    print(f"Chunk size: {chunk_size}")
    
    try:
        # Cargar configuración
        config_file = os.path.join(os.path.dirname(__file__), "..", "config.yml")
        if not os.path.exists(config_file):
            config_file = "config.yml"  # Fallback
            
        with open(config_file, "r") as conf:
            config = yaml.safe_load(conf)
        
        # Actualizar configuración para esta operación
        config["collection_name"] = collection
        config["chunk_size"] = chunk_size
        config["data_path"] = directory
        
        # Crear instancia de Data
        data_processor = Data(config)
        
        # Verificar si hay archivos PDF para procesar con docling
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
        
        if pdf_files:
            print(f"📄 Encontrados {len(pdf_files)} archivos PDF. Procesando con docling...")
            data_processor.docling()
            print("✅ Procesamiento con docling completado")
        else:
            print("ℹ️ No se encontraron archivos PDF para procesar")
        
        # Verificar archivos de texto disponibles para indexar
        all_files = os.listdir(directory)
        text_files = [f for f in all_files if f.endswith(('.txt', '.md'))]
        
        print(f"📝 Encontrados {len(text_files)} archivos de texto para indexar")
        print(f"Archivos disponibles: {all_files}")
        
        if len(text_files) == 0:
            raise Exception("No se encontraron archivos de texto para indexar. Verifica que docling haya procesado correctamente los PDFs.")
        
        # Inicializar embeddings y cache
        cache_manager = get_cache_manager()
        cache_manager.ensure_embedding_model_ready(config)
        embed_model = cache_manager.get_cached_embedding_model()
        
        if embed_model is None:
            embed_model = cache_manager.create_embedding_model(config)
        
        # Crear LLM (necesario para la indexación)
        from llama_index.llms.ollama import Ollama
        llm = Ollama(model=config["llm_name"], base_url=config["llm_url"])
        
        print("🔄 Iniciando indexación en Qdrant...")
        
        # Procesar con extensión .txt (incluye archivos _docling.txt)
        index = data_processor.ingest(embedder=embed_model, llm=llm, extension=".txt")
        
        print(f"✅ Documentos procesados e indexados correctamente en la colección '{collection}'")
        return True
        
    except Exception as e:
        print(f"❌ Error procesando documentos: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

def main():
    parser = argparse.ArgumentParser(description="Procesar documentos subidos y cargarlos en Qdrant")
    parser.add_argument("--directory", required=True, help="Directorio con los documentos a procesar")
    parser.add_argument("--collection", required=True, help="Nombre de la colección en Qdrant")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Tamaño de los chunks")
    
    args = parser.parse_args()
    
    try:
        process_documents_for_upload(args.directory, args.collection, args.chunk_size)
        print("🎉 Procesamiento completado exitosamente")
        return 0
    except Exception as e:
        print(f"💥 Error en el procesamiento: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())