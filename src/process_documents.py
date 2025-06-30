#!/usr/bin/env python3

# ===== CONFIGURACIÓN AUTOMÁTICA DE CACHE =====
from cache_manager import initialize_cache, get_cache_manager

# Configurar cache antes de importar FastEmbed
cache_info = initialize_cache()
cache_manager = get_cache_manager()

import os
import yaml
from data import Data

def process_documents_for_upload(directory, collection, chunk_size=1024):
    """Procesa documentos subidos desde el frontend - NO necesita LLM."""
    print("=== PROCESANDO DOCUMENTOS SUBIDOS ===")
    print(f"Directorio: {directory}")
    print(f"Colección: {collection}")
    print(f"Chunk size: {chunk_size}")
    
    try:
        config_file = os.path.join(os.path.dirname(__file__), "config.yml")
        with open(config_file, "r") as conf:
            config = yaml.safe_load(conf)
        
        # Cambiar temporalmente el data_path
        config["data_path"] = directory
        
        data = Data(config)
        
        # 1. Procesar PDFs con docling
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
        if pdf_files:
            print(f"📄 Encontrados {len(pdf_files)} archivos PDF. Procesando con docling...")
            data.docling()
            print("✅ Procesamiento con docling completado")
        
        # 2. Indexar archivos de texto
        txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
        if txt_files:
            print(f"📝 Encontrados {len(txt_files)} archivos de texto para indexar")
            print(f"Archivos disponibles: {os.listdir(directory)}")
            
            # Solo crear modelo de embeddings - NO LLM
            embed_model = cache_manager.create_embedding_model(config)
            
            # Indexar en Qdrant - SIN LLM
            data.ingest(
                embedder=embed_model,
                extension=".txt",
                collection_name=collection,
                chunk_size=chunk_size
            )
            print("✅ Indexación completada")
        else:
            print("❌ No se encontraron archivos de texto para indexar")
            
    except Exception as e:
        print(f"❌ Error procesando documentos: {str(e)}")
        raise Exception(f"Error en procesamiento: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, help="Directorio con documentos")
    parser.add_argument("--collection", required=True, help="Colección de Qdrant")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Tamaño de chunk")
    # ❌ ELIMINAR: NO necesita --model para indexación
    
    args = parser.parse_args()
    
    process_documents_for_upload(
        directory=args.directory,
        collection=args.collection,
        chunk_size=args.chunk_size
    )