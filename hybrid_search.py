import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from config import TOP_K, BM25_WEIGHT


class BM25Retriever:
  def __init__(self,chunks):
    self.chunks=chunks
    self.tokenized_corpus=[jieba.lcut(doc) for doc in chunks]
    self.bm25=BM25Okapi(self.tokenized_corpus)

  def search(self,question):
    tokenized_query = jieba.lcut(question)
    score = self.bm25.get_scores(tokenized_query)
    score = np.array(score)
    max_score = score.max()
    min_score = score.min()

    if max_score == min_score:
      return np.zeros_like(score)
    bm25_score_normalized = (score-min_score)/(max_score-min_score)
    return bm25_score_normalized


class VectorRetriever:
  def __init__(self,vector_store):
    self.vector_store = vector_store #  因为后面要调用 vector_store.py里面的函数，这里是在 RAGpipeline 流程里注入的

  def search(self,question):
    query_embedding = self.vector_store.embed_query(question)
    # get_all_docs_and_vectors() 返回 (documents, embeddings) 元组，这里只需要向量
    _, doc_embedding = self.vector_store.get_all_docs_and_vectors()

    query_embedding = np.array(query_embedding)
    doc_embedding = np.array(doc_embedding)

    distance = np.linalg.norm(query_embedding-doc_embedding, axis=1)
    max_d = distance.max()
    min_d = distance.min()

    if max_d == min_d:
      return np.ones_like(distance)
    vector_score_normalized = 1 - (distance-min_d)/(max_d-min_d)
    return vector_score_normalized


class HybridRetriever:
  def __init__(self, chunks, vector_store, bm25_weight=BM25_WEIGHT):
    self.chunks = chunks
    self.vector_store = vector_store
    self.bm25_weight = bm25_weight

    self.bm25_retriever = BM25Retriever(chunks)
    self.vector_retriever = VectorRetriever(vector_store)


  def search(self,question, top_k=TOP_K):
    bm25_score = self.bm25_retriever.search(question)
    vector_score = self.vector_retriever.search(question)

    combined_score = (self.bm25_weight*bm25_score + (1-self.bm25_weight)*vector_score)

    top_index = combined_score.argsort()[::-1][:top_k]

    hybrid_result = [self.chunks[i] for i in top_index]

    return hybrid_result