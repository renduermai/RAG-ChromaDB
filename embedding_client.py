from config import BATCH_SIZE
from models import get_ali_embeddings


class EmbeddingClient:

    def __init__(self,batch_size=BATCH_SIZE):
        self.batch_size = batch_size
        self.embedding_model = get_ali_embeddings()


    def embed_documents(self,texts):
        all_vectors = []

        for i in range(0,len(texts),self.batch_size):
            batch = texts[i:i+self.batch_size]
            vectors = self.embedding_model.embed_documents(batch)
            all_vectors.extend(vectors)

        return all_vectors

    def embed_query(self,query):
        vector = self.embedding_model.embed_query(query)
        return vector
