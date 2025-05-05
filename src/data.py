from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import Document
import os
import argparse
import yaml
import qdrant_client

# Importaciones de docling
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions


class Data:
    def __init__(self, config):
        self.config = config

    def docling(self):
        data_path = self.config["data_path"]
        pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith(".pdf")]

        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.do_cell_matching = False
        pipeline_options.do_ocr = True

        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        for pdf_file in pdf_files:
            source = os.path.join(data_path, pdf_file)
            try:
                result = doc_converter.convert(source)
                output_file = os.path.join(data_path, pdf_file.replace('.pdf', '_docling.txt'))
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result.document.export_to_markdown())
                print(f"[EXITO] Texto limpio guardado en '{output_file}'")
            except Exception as e:
                print(f"[ERROR] Falló la conversión de '{pdf_file}': {e}")

    def ingest(self, embedder, llm, extension=".txt"):
        print(f"Indexing data with extension '{extension}'...")
        data_path = self.config["data_path"]
        print(f"Reading files from directory: {data_path}")
        files_to_index = [
            f for f in os.listdir(data_path) if f.endswith(extension)
        ]
        print(f"Found {len(files_to_index)} files to index.")

        documents = []
        for file in files_to_index:
            file_path = os.path.join(data_path, file)
            print(f"Reading file: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append(Document(text=content, doc_id=file, metadata={"file_path": data_path}))
            except Exception as e:
                print(f"[ERROR] No se pudo leer el archivo {file}: {e}")
                continue
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
        print(f"Data indexed successfully to Qdrant. Collection: {self.config['collection_name']}")
        return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--ingest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ingest data to Qdrant vector Database."
    )
    parser.add_argument(
        "-d", "--docling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preprocess PDF documents using Docling and save as .txt."
    )
    parser.add_argument(
        "-e", "--extension",
        type=str,
        default=".txt",
        help="File extension to index in Qdrant (default: .txt)."
    )

    args = parser.parse_args()
    config_file = "config.yml"
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)
    data = Data(config)

    if args.docling:
        print("Preprocessing documents with Docling...")
        data.docling()

    if args.ingest:
        print("Loading Embedder...")
        print("Initializing FastEmbedEmbedding model. This may take a while if it's the first time...")
        embed_model = FastEmbedEmbedding(model_name=config["embedding_model"])
        print("FastEmbedEmbedding model loaded successfully.")
        llm = Ollama(model=config["llm_name"], base_url=config["llm_url"])
        data.ingest(embedder=embed_model, llm=llm, extension=args.extension)
