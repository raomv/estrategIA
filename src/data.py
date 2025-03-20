from llama_index.core import StorageContext, VectorStoreIndex  # Importaciones desde el core
from llama_index.core.settings import Settings  # Para configuraciones globales
from llama_index.core import SimpleDirectoryReader  # Importación actualizada
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding  # FastEmbed
from llama_index.llms.ollama import Ollama  # Ollama LLM
import os
import argparse
import yaml
import qdrant_client


class Data:
    def __init__(self, config):
        self.config = config

    def ingest(self, embedder, llm):
        print("Indexing data...")
        documents = SimpleDirectoryReader(self.config["data_path"]).load_data()

        client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=self.config["collection_name"]
        )
        storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)

        # Configuración global usando Settings
        Settings.embed_model = embedder
        Settings.llm = llm
        Settings.chunk_size = self.config["chunk_size"]

        # Crear el índice sin ServiceContext
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

    args = parser.parse_args()
    config_file = "config.yml"
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)
    data = Data(config)
    if args.ingest:
        print("Loading Embedder...")
        # Usar FastEmbedEmbedding
        embed_model = FastEmbedEmbedding(model_name=config["embedding_model"])
        llm = Ollama(model=config["llm_name"], base_url=config["llm_url"])
        data.ingest(embedder=embed_model, llm=llm)