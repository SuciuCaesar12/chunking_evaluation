from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    
    def __init__(
        self, 
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
        device: str = 'cpu' # just for sake of consistency, we didn't install pytorch 
                            # since the dataset is small and running on CPU is fast enough for this use case
    ):
        self.model = SentenceTransformer(model_name, device=device)
    
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input)
