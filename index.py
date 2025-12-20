import logging
import os

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank

from settings import PERSIST_DIR, embed_model

logging.info("Initializing vector storage")

if os.path.exists(PERSIST_DIR):
    logging.info("Storage exists so loading index")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
else:
    logging.info("Building index")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents=documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)


def get_retriever(top_k: int = 10) -> BaseRetriever:
    logging.info("Creating a retriever for the index")
    return index.as_retriever(similarity_top_k=top_k)


def get_query_engine(top_k: int = 3, response_mode: str = "compact") -> BaseQueryEngine:
    logging.info("Creating a query engine for the index")
    return index.as_query_engine(similarity_top_k=top_k, response_mode=response_mode)


def get_ranker() -> SentenceTransformerRerank:
    return SentenceTransformerRerank(top_n=5, device="cpu")
