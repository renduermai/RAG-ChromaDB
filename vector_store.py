import chromadb
from langchain_chroma import Chroma # 修改代码（langchian）

from config import TOP_K, CHROMA_PATH, COLLECTION_NAME, BATCH_SIZE
from chromadb import Settings

from models import get_ali_embeddings


class VectorStore:
    # 原代码
    # def __init__(self,persist_path=CHROMA_PATH, space="l2"):
    #     if persist_path:
    #         self.client = chromadb.PersistentClient(path=persist_path)
    #     else:
    #         self.client = chromadb.Client(Settings(allow_reset=True))
    #
    #     self.collection = self.client.get_or_create_collection(
    #         name=COLLECTION_NAME,
    #         metadata={"hnsw:space": space}
    #     )

    # 修改代码(langchain)
    def __init__(self, persist_path=CHROMA_PATH,space="l2"):
        self.store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_ali_embeddings(),
            persist_directory=persist_path,
            collection_metadata={"hnsw:space": space}
        )



    # 原代码
    # def add_documents(self, documents, vectors, ids=None):
    #     if ids is None:
    #         ids = [f"id_{i}" for i in range(len(documents))]
    #
    #     self.collection.add(
    #         ids=ids,
    #         documents=documents,
    #         embeddings=vectors
    #     )

    # 修改代码(langchain)
    def add_documents(self, documents, ids=None):
        if ids is None:
            ids = [f"id_{i}" for i in range(len(documents))]

        for i in range(0,len(documents),BATCH_SIZE):
            self.store.add_texts(
                documents[i:i + BATCH_SIZE],
                ids=ids[i:i + BATCH_SIZE])

    # 原代码
    # def search(self, query_vector, top_k=TOP_K):
    #     results = self.collection.query(
    #         query_embeddings=query_vector,
    #         n_results=TOP_K
    #     )
    #     return results

    # 修改代码(langchain)
    def search(self, query, top_k=TOP_K):
        docs = self.store.similarity_search(query, k=TOP_K)
        return [doc.page_content for doc in docs]


