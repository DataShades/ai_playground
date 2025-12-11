from pathlib import Path
from time import time

from sqlalchemy import make_url
import click
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import textwrap

from ai_pg.config import Config


@click.group()
def rag():
    """RAG commands for indexing and querying documents in PostgreSQL vector store."""
    pass


@rag.command(name="index-documents")
def index_documents():
    """Index documents into PostgreSQL vector store."""
    # Replace default embedding model with Ollama embedding
    Settings.embed_model = OllamaEmbedding(
        model_name=Config.OLLAMA_EMBEDDING_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
    )

    # Load documents from the 'data' directory
    documents = SimpleDirectoryReader(
        Path(__file__).parent / ".." / ".." / "data",
        filename_as_id=True,
    ).load_data()

    # Create and persist the index in PostgreSQL vector store
    VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(vector_store=get_vector_store()),
        show_progress=True,
    )


@rag.command(name="index-document")
@click.argument("filename", type=str)
def index_document(filename: str) -> None:
    """Index a specific document into PostgreSQL vector store.

    Removes existing index for the document before re-indexing.

    Args:
        filename (str): The name of the file to index, located in the 'data' directory.
    """
    # Replace default embedding model with Ollama embedding
    Settings.embed_model = OllamaEmbedding(
        model_name=Config.OLLAMA_EMBEDDING_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
    )

    # Load a specific document from the 'data' directory
    # TODO: filename_as_id reconsidered when we're going to work with CKAN resources
    documents = SimpleDirectoryReader(
        Path(__file__).parent / ".." / ".." / "data", filename_as_id=True
    ).load_data()
    file_path = str(Path(__file__).parent / ".." / ".." / "data" / filename)

    vector_store = get_vector_store()

    for doc in documents:
        if doc.metadata.get("file_path") != file_path:
            continue

        # Remove existing index and re-index the document
        vector_store.delete(ref_doc_id=doc.doc_id)

        VectorStoreIndex.from_documents(
            [doc],
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            show_progress=True,
        )

        break


@rag.command
@click.argument("query", type=str)
def query_index(query: str):
    # Replace default embedding model with Ollama embedding
    Settings.embed_model = OllamaEmbedding(
        model_name=Config.OLLAMA_EMBEDDING_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
    )

    # Replace default OpenAI LLM with Ollama LLM
    Settings.llm = Ollama(
        base_url=Config.OLLAMA_BASE_URL,
        model=Config.OLLAMA_MODEL,
        context_window=Config.OLLAMA_CONTEXT_WINDOW,
        request_timeout=Config.OLLAMA_TIMEOUT,
        thinking=Config.OLLAMA_THINKING,
        temperature=Config.OLLAMA_TEMPERATURE,
    )

    # Load existing data instead of re-indexing
    index = VectorStoreIndex.from_vector_store(get_vector_store())
    query_engine = index.as_query_engine()

    start = time()
    response = query_engine.query(query)
    total_time = time() - start

    click.secho(
        f"Completed request in {total_time:.2f} seconds \n", fg="yellow"
    )

    click.secho(textwrap.fill(str(response), width=80), fg="green")


def get_vector_store() -> PGVectorStore:
    url = make_url(Config.PG_URI)

    vector_store = PGVectorStore.from_params(
        database=Config.PG_DB_NAME,
        host=url.host,
        password=url.password,
        port=str(url.port or 5432),
        user=url.username,
        table_name=Config.PG_TABLE_NAME,
        embed_dim=Config.OLLAMA_EMBEDDING_MODEL_DIM,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    return vector_store
