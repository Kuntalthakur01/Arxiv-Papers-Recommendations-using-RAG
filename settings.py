import os

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# EMBEDDING_MODEL = HuggingFaceEmbedding(model_name="Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
EMBEDDING_MODEL = OpenAIEmbedding(model="text-embedding-3-large")
DATASET_PATH = os.path.abspath("./2024_arxiv_papers_CS/")
PERSIST_PATH = os.path.abspath("./storage")
TEXT_VECTOR_STORE_PATH = os.path.join(PERSIST_PATH, "default__vector_store.json")
IMAGE_VECTOR_STORE_PATH = os.path.join(PERSIST_PATH, "image__vector_store.json")

# DynamoDB tables
DOCSTORE_TABLE_NAME = "docstore"
INDEX_STORE_TABLE_NAME = "index_store"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_ID = "multimodal-vector-store-index"
