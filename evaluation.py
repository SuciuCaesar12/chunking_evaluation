from pathlib import Path

from utils.chunking import *
from utils.data import *
from utils.chromadb import *
from metrics import *
from chunking import BaseChunker
from embeddings import EmbeddingFunction


class Evaluation:

    def __init__(
        self, 
        questions_csv_path: Path, 
        chroma_client: ChromaDBManager,
        corpora_path: Path, 
        embedding_function: EmbeddingFunction,
        metrics: List[str] = ['recall', 'precision', 'iou'],
        batch_size: int = 500,
    ):  
        self.corpora_path = corpora_path
        self.questions_csv_path = questions_csv_path
        
        self.metrics = metrics
        
        self.chroma_client = chroma_client
        self.embedding_function = embedding_function
        self.batch_size = batch_size
        
        self.load_question_in_collection_()
        
    
    def get_chunks_and_metadata_from_corpus_(self, chunker: BaseChunker):
        corpora = load_corpora(self.corpora_path)
        docs, metadatas = chunker.split_text(corpora), []
        
        for doc in docs:
            try:
                _, start_index, end_index = rigorous_document_search(corpora, doc)
            except:
                raise Exception(f"Error in finding {doc}")
            
            metadatas.append({"start_index": start_index, "end_index": end_index})
        
        return docs, metadatas


    def load_question_in_collection_(self):
        
        questions_df = load_questions_df(
            self.questions_csv_path, corpora_id=self.corpora_path.stem
        )
        
        # read it once and used it for any suqsequent calculations of metrics
        self.target_metadatas = [
            [
                {
                    'start_index': int(excerpt['start_index']), 
                    'end_index': int(excerpt['end_index'])
                } 
                for excerpt in ref
            ]
            for ref in questions_df['references'].tolist()
        ]
        
        status, collection = self.chroma_client.get_or_create_collection(
            name=f"question_embeddings", embedding_function=self.embedding_function
        )
        if status == CollectionStatus.LOADED:
            self.questions_collection = collection
            return

        self.chroma_client.add(
            collection, 
            ids=questions_df.index.astype(str).tolist(), 
            documents=questions_df['question'].tolist(), 
            batch_size=self.batch_size
        )

        self.questions_collection = collection


    def load_chunked_corpora_in_collection_(self, chunker: BaseChunker):
        
        # compatible with fixed-size chunkers only, but for our use case, it's fine
        collection_name = (
            f"{self.corpora_path.stem}" +
            f"_size_{chunker._chunk_size}" +
            f"_overlap_{chunker._chunk_overlap}" 
        )
        status, collection = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )
        if status == CollectionStatus.LOADED:
            return collection

        docs, metadatas = self.get_chunks_and_metadata_from_corpus_(chunker)
        self.chroma_client.add(
            collection,
            ids=[str(i) for i in range(len(docs))],
            documents=docs,
            metadatas=metadatas,
            batch_size=self.batch_size
        )

        return collection
    
    def run(self, chunker: BaseChunker, num_chunks: int):
        
        embeddings = self.questions_collection.get(include=['embeddings'])['embeddings']
        chunks_collection = self.load_chunked_corpora_in_collection_(chunker)
        
        retrieved_metadatas = chunks_collection.query(embeddings, n_results=num_chunks)['metadatas']
        
        metric_resuls = calculate_metrics(retrieved_metadatas, self.target_metadatas)
        metric_resuls['num_chunks'] = num_chunks
        metric_resuls['chunk_size'] = chunker._chunk_size
        metric_resuls['chunk_overlap'] = chunker._chunk_overlap
        
        return metric_resuls
