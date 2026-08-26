import chromadb
from config import TOP_K, CHROMA_PATH, COLLECTION_NAME
from chromadb import Settings

class VectorStore:

    def __init__(self,persist_path=CHROMA_PATH, space="l2"):
        if persist_path:
            self.client = chromadb.PersistentClient(path=persist_path)
        else:
            self.client = chromadb.Client(Settings(allow_reset=True))

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": space}
        )

    def add_documents(self, documents, vectors, ids=None):
        if ids is None:
            ids = [f"id_{i}" for i in range(len(documents))]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors
        )

    def search(self, query_vector, top_k=TOP_K):
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=TOP_K
        )
        return results

