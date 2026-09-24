import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from config import TOP_K, BM25_WEIGHT


class BM25Retriever:
    """BM25 全文检索模块：基于关键词匹配的检索。"""

    def __init__(self, chunks):
        """
        初始化 BM25 检索器。

        解决什么问题：
        - 对全部语料分词，建立 BM25 统计索引，只做一次，查询时复用。

        输入：
        - chunks：切好块的文档列表。

        输出：
        - self.chunks：保存语料，用于最后按索引取回原文。
        - self.bm25：建好的 BM25 索引。

        用了什么模块：
        - jieba.lcut()：中文分词。
        - BM25Okapi：BM25 算法实现。
        """
        self.chunks = chunks

        # 文档分词：BM25 需要把文本拆成词，才能统计词频等信息
        self.tokenized_corpus = [jieba.lcut(doc) for doc in chunks]

        # 初始化 BM25 对象，一次性算好全部文档的统计信息
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query):
        """
        BM25 检索：返回问题对全部语料的归一化分数。

        解决什么问题：
        - 问题分词，和全部文档算 BM25 相似度分数。
        - 分数归一化到 [0, 1]，方便和向量分数加权融合。

        输入：
        - query：用户问题。

        输出：
        - bm25_scores_normalized：numpy 数组，每个元素是语料中对应文档的 [0, 1] 分数。

        用了什么模块：
        - jieba.lcut()：问题分词。
        - self.bm25.get_scores()：BM25 算查询词与每个文档的相似度分数。
        - np.array + max/min：归一化到 [0, 1]。
        """
        # 问题分词
        tokenized_query = jieba.lcut(query)

        # BM25 打分
        scores = np.array(self.bm25.get_scores(tokenized_query))

        # 归一化到 [0, 1]：(分数 - 最低分) / (最高分 - 最低分)
        max_score = scores.max()
        min_score = scores.min()

        if max_score == min_score:
            # 所有文档分数相同，返回全 0，避免除零
            return np.zeros_like(scores)

        bm25_scores_normalized = (scores - min_score) / (max_score - min_score)
        return bm25_scores_normalized


class VectorRetriever:
    """向量检索打分模块：负责问题向量化、读取文档向量、计算归一化相似度。"""

    def __init__(self, vector_store):
        """
        初始化向量检索打分器。

        解决什么问题：
        - 持有向量数据库对象，用于问题向量化和读取全量文档向量。

        输入：
        - vector_store：向量数据库对象。

        输出：
        - self.vector_store：向量数据库对象。
        """
        self.vector_store = vector_store

    def search_scores(self, question):
        """
        向量检索打分：返回问题对全部语料的归一化相似度分数。

        解决什么问题：
        - 问题向量化。
        - 问题向量和全部文档向量算 l2 欧氏距离。
        - 距离越小越相似，转成 [0, 1] 相似度分数。

        输入：
        - question：用户问题。

        输出：
        - vector_scores_normalized：numpy 数组，每个元素是语料中对应文档的 [0, 1] 相似度。

        用了什么模块：
        - self.vector_store.embed_query()：问题向量化。
        - self.vector_store.get_all_docs_and_vectors()：读全量文档向量。
        - np.linalg.norm(..., axis=1)：按行算 l2 距离。
        """
        # 问题向量化
        query_embedding = np.array(self.vector_store.embed_query(question))

        # 读全部文档文本和向量，此处不重复调 embedding API
        # documents 当前打分逻辑用不到，所以用 _documents 接收
        _documents, doc_embeddings = self.vector_store.get_all_docs_and_vectors()
        doc_embeddings = np.array(doc_embeddings)

        # 算 l2 距离：距离越小越相似
        distances = np.linalg.norm(query_embedding - doc_embeddings, axis=-1)

        # 距离转相似度并归一化：1 - (距离 - 最小距离) / (最大距离 - 最小距离)
        max_d = distances.max()
        min_d = distances.min()

        if max_d == min_d:
            # 极端情况，所有向量距离相同，返回全 1
            return np.ones_like(distances)
        # 注意这里和 bm25不一样，l2是越小越好
        vector_scores_normalized = 1 - (distances - min_d) / (max_d - min_d)
        return vector_scores_normalized


class HybridRetriever:
    """混合检索器：BM25 分数 + 向量相似度分数，加权融合后取 top_k。"""

    def __init__(self, chunks, vector_store, bm25_weight=BM25_WEIGHT):
        """
        初始化混合检索器。

        解决什么问题：
        - 备好 BM25 检索器。
        - 备好向量检索打分器。

        输入：
        - chunks：切好块的文档列表，BM25 语料，与入库语料是同一份，顺序一致。
        - vector_store：向量数据库对象。
        - bm25_weight：BM25 融合权重，来自 config。

        输出：
        - self.chunks / self.bm25_retriever / self.vector_retriever / self.bm25_weight。
        """
        self.chunks = chunks
        self.bm25_weight = bm25_weight

        # 建 BM25 索引
        self.bm25_retriever = BM25Retriever(chunks)

        # 建向量检索打分器
        self.vector_retriever = VectorRetriever(vector_store)

    def search(self, question, top_k=TOP_K):
        """
        混合检索入口：融合 BM25 和向量分数，返回 top_k 个文档。

        解决什么问题：
        - 拿到 BM25 归一化分数和向量归一化分数。
        - 加权融合：combined = BM25_WEIGHT * bm25 + (1 - BM25_WEIGHT) * vector。
        - 按 combined 降序排序，取 top_k 个文档原文。

        输入：
        - question：用户问题。
        - top_k：返回的文档数，来自 config。

        输出：
        - hybrid_results：融合排序后最相关的 top_k 个文档文本的列表。

        用了什么模块：
        - self.bm25_retriever.search()：BM25 打分。
        - self.vector_retriever.search_scores()：向量打分。
        - combined_scores.argsort()[::-1]：降序排序取索引。
        """
        # BM25 分数，已归一化
        bm25_scores = self.bm25_retriever.search(question)

        # 向量相似度分数，已归一化
        vector_scores = self.vector_retriever.search_scores(question)

        # 加权融合
        combined_scores = (
            self.bm25_weight * bm25_scores
            + (1 - self.bm25_weight) * vector_scores
        )

        # 降序排序，取 top_k 个索引
        top_index = combined_scores.argsort()[::-1][:top_k]

        # 按索引取回文档原文
        hybrid_results = [self.chunks[i] for i in top_index]
        return hybrid_results