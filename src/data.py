from llama_index.core import StorageContext, VectorStoreIndex  # Importaciones desde el core
from llama_index.core.settings import Settings  # Para configuraciones globales
from llama_index.core import SimpleDirectoryReader  # Importación actualizada
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding  # FastEmbed
from llama_index.llms.ollama import Ollama  # Ollama LLM
from llama_index.core.schema import Document  # Importación para Document
import os
import argparse
import yaml
import qdrant_client


class Data:
    def __init__(self, config):
        self.config = config

    def ingest(self, embedder, llm, extension=".txt"):
        print(f"Indexing data with extension '{extension}'...")
        data_path = self.config["data_path"]
        print(f"Reading files from directory: {data_path}")
        files_to_index = [
            f for f in os.listdir(data_path) if f.endswith(extension)
        ]  # Filtrar archivos por extensión
        print(f"Found {len(files_to_index)} files to index.")

        documents = []
        for file in files_to_index:
            file_path = os.path.join(data_path, file)
            print(f"Reading file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Crear instancias de Document en lugar de diccionarios
                documents.append(Document(text=content, doc_id=file, metadata={"file_path": data_path}))
        print(f"Loaded {len(documents)} documents.")

        print("Connecting to Qdrant...")
        client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=self.config["collection_name"]
        )
        storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)

        print("Configuring global settings for embeddings and LLM...")
        Settings.embed_model = embedder
        Settings.llm = llm
        Settings.chunk_size = self.config["chunk_size"]

        print("Indexing documents in Qdrant...")
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print(
            f"Data indexed successfully to Qdrant. Collection: {self.config['collection_name']}"
        )
        return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--ingest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ingest data to Qdrant vector Database.",
    )
    parser.add_argument(
        "-d",
        "--docling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preprocess documents and save clean text.",
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        default=".txt",
        help="File extension to index in Qdrant (default: .txt).",
    )

    args = parser.parse_args()
    config_file = "config.yml"
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)
    data = Data(config)

    if args.docling:
        print("Preprocessing documents with docling...")
        data.docling()

    if args.ingest:
        print("Loading Embedder...")
        # Mensaje antes de inicializar el modelo de embeddings
        print("Initializing FastEmbedEmbedding model. This may take a while if it's the first time...")
        embed_model = FastEmbedEmbedding(model_name=config["embedding_model"])
        print("FastEmbedEmbedding model loaded successfully.")
        llm = Ollama(model=config["llm_name"], base_url=config["llm_url"])
        data.ingest(embedder=embed_model, llm=llm, extension=args.extension)