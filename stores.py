import os

import faiss
from llama_index.core import StorageContext
from llama_index.storage.docstore.dynamodb import DynamoDBDocumentStore
from llama_index.storage.index_store.dynamodb import DynamoDBIndexStore
from llama_index.vector_stores.faiss import FaissVectorStore

from settings import (
    DOCSTORE_TABLE_NAME,
    IMAGE_VECTOR_STORE_PATH,
    INDEX_STORE_TABLE_NAME,
    PERSIST_PATH,
    TEXT_VECTOR_STORE_PATH,
)


def init_storage_context() -> StorageContext:
    text_index = faiss.IndexFlatL2(384)
    image_index = faiss.IndexFlatL2(512)
    docstore = DynamoDBDocumentStore.from_table_name(DOCSTORE_TABLE_NAME)
    index_store = DynamoDBIndexStore.from_table_name(INDEX_STORE_TABLE_NAME)
    text_store = FaissVectorStore(
        faiss_index=text_index,
    )
    image_store = FaissVectorStore(faiss_index=image_index)

    if os.path.exists(PERSIST_PATH):
        return StorageContext.from_defaults(
            vector_store=FaissVectorStore.from_persist_path(TEXT_VECTOR_STORE_PATH),
            image_store=FaissVectorStore.from_persist_path(IMAGE_VECTOR_STORE_PATH),
            docstore=docstore,
            index_store=index_store,
            persist_dir=PERSIST_PATH,
        )
    else:
        return StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store,
            vector_store=text_store,
            image_store=image_store,
        )
