from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import qdrant_client
import yaml

class RAG:
    def __init__(self, config_file, llm):
        self.config = config_file
        self.qdrant_client = qdrant_client.QdrantClient(
            url=self.config['qdrant_url']
        )
        self.llm = llm  # ollama llm
    
    def load_embedder(self):
        # Usar FastEmbedEmbedding
        embed_model = FastEmbedEmbedding(model_name=self.config['embedding_model'])
        return embed_model

    def qdrant_index(self):
        client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=self.config['collection_name']
        )

        # Configuración global usando Settings
        Settings.llm = self.llm  # Configura el LLM
        Settings.embed_model = self.load_embedder()  # Configura el modelo de embeddings

        # Crear el índice conectándose a la colección existente
        index = VectorStoreIndex.from_vector_store(
            vector_store=qdrant_vector_store
        )
        return index