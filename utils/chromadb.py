from pathlib import Path
from typing import Optional, Union
from enum import Enum
from chromadb import PersistentClient, Client, Collection, EmbeddingFunction

    
class CollectionStatus(Enum):
    CREATED = 0
    LOADED = 1


class ChromaDBManager:
    def __init__(self, db_path: Optional[Path] = None, verbose: bool = True):
        self.client_ = PersistentClient(path=str(db_path)) if db_path else Client()
        self.verbose = verbose

    def get_or_create_collection(
        self, 
        name: str, 
        embedding_function: Optional[EmbeddingFunction] = None,
        metadata: dict = {"hnsw:search_ef":50},
    ):
        exists = name in self.client_.list_collections()
        assert exists or embedding_function is not None,\
            "Embedding function must be provided for new collections"
        status = CollectionStatus.LOADED if exists else CollectionStatus.CREATED
        
        collection = self.client_.get_or_create_collection(
            name, embedding_function=embedding_function, metadata=metadata
        )
        return status, collection
    
    def add(
        self,
        collection: Union[str, Collection],
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] = None,
        batch_size: int = 500
    ):
        if isinstance(collection, str):
            collection = self.client_.get_collection(collection)
        
        for step in range(0, len(ids), batch_size):
            start, end = step, step + batch_size
            collection.add(
                ids=ids[start: end],
                documents=documents[start: end],
                metadatas=metadatas[start: end] if metadatas else None,
            )
