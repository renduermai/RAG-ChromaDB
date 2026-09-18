# hybrid_search.py
# 混合检索模块
#
# 解决什么问题:
#  - 纯向量检索对"关键词精确匹配"(如专有名词、型号、错字)不敏感
#  - 把BM25全文检索(关键词匹配)和向量相似度检索(语义匹配)加权融合
#    兼顾两者优点,取更准确的top_k文档
#
# 核心思路(与原"混合检索的代码.txt"一致):
#  - BM25对全部语料打分,归一化到[0,1]
#  - 向量用l2距离对全部语料打分,转成相似度并归一化到[0,1]
#  - combined = BM25_WEIGHT * bm25分数 + (1-BM25_WEIGHT) * 向量分数
#  - 按combined降序取top_k
#
# 与原版的不同点(刻意简化,保证跑通):
#  - 向量不再每次查询重新调embedding API,而是从vector_store里读已有向量

import numpy as np
from rank_bm25 import BM25Okapi
import jieba

from config import TOP_K, BM25_WEIGHT

import re

def tokenize(text):
    "统一分词函数:语料和问题必须走同一套规则,词面才能对上"
    # 1.英文统一转小写:jieba只负责切词不管大小写,
    #   不加这句,语料里的 "DeepSeek" 和问题里的 "deepseek" 是两个不同的词
    text = text.lower()
    tokens = jieba.lcut(text)
    # 2.中文二字兜底:人名等新词在问题/语料里jieba切法可能不一致,
    #   补充相邻二字组合(周鸿/鸿祎),保证词面匹配不丢
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens

class BM25Retriever:
    "BM25全文检索模块:基于关键词匹配的检索"

    def __init__(self, chunks):
        "初始化BM25检索器"
        """
        解决什么问题:
         - 对全部语料分词,建立BM25统计索引(只做一次,查询时复用)

        输入:
         - chunks:切好块的文档列表(来自document_loader.splitter)

        输出:
         - self.chunks:保存语料,用于最后按索引取回原文
         - self.bm25:建好的BM25索引

        用了什么模块:
         - jieba.lcut():中文分词
         - BM25Okapi:BM25算法实现,from rank_bm25 import BM25Okapi
        """
        self.chunks = chunks
        # 文档分词:BM25需要把文本拆成词,才能统计词频等信息
        # self.tokenized_corpus = [jieba.lcut(doc) for doc in chunks]
        self.tokenized_corpus = [tokenize(doc) for doc in chunks]
        # 初始化BM25对象,一次性算好全部文档的统计信息
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query):
        "BM25检索:返回问题对全部语料的归一化分数"
        """
        解决什么问题:
         - 问题分词,和全部文档算BM25相似度分数
         - 分数归一化到[0,1],方便和向量分数加权融合

        输入:
         - query:用户问题

        输出:
         - bm25_scores_normalized: numpy数组,每个元素是语料中对应文档的[0,1]分数

        用了什么模块:
         - jieba.lcut():问题分词
         - self.bm25.get_scores():BM25算查询词与每个文档的相似度分数
         - np.array + max/min:归一化到[0,1]
        """
        # 问题分词
        # tokenized_query = jieba.lcut(query)
        tokenized_query = tokenize(query)
        # BM25打分
        scores = np.array(self.bm25.get_scores(tokenized_query))
        # 归一化到[0,1]: (分数-最低分)/(最高分-最低分)
        max_score = scores.max()
        min_score = scores.min()
        if max_score == min_score:
            # 所有文档分数相同(极端情况),返回全0,避免除零
            return np.zeros_like(scores)
        bm25_scores_normalized = (scores - min_score) / (max_score - min_score)
        return bm25_scores_normalized


class HybridRetriever:
    "混合检索器:BM25分数 + 向量相似度分数,加权融合后取top_k"

    def __init__(self, chunks, vector_store, bm25_weight=BM25_WEIGHT):
        "初始化混合检索器"
        """
        解决什么问题:
         - 备好BM25检索器(本地索引)
         - 备好向量数据来源(vector_store)

        输入:
         - chunks:切好块的文档列表(BM25语料,与入库的语料是同一份,顺序一致)
         - vector_store:向量数据库对象(用于读全量文档向量)
         - bm25_weight:BM25融合权重,来自config

        输出:
         - self.chunks / self.bm25_retriever / self.vector_store / self.bm25_weight

        用了什么模块:
         - BM25Retriever:本模块的BM25检索器
         - vector_store:项目原有的向量数据库对象(调用其新增的两个方法)
        """
        self.chunks = chunks
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        # 建BM25索引
        self.bm25_retriever = BM25Retriever(chunks)

    def vector_search_scores(self, question):
        "向量检索打分:返回问题对全部语料的归一化相似度分数"
        """
        解决什么问题:
         - 问题向量化
         - 问题向量和全部文档向量算l2(欧氏)距离
         - 距离越小越相似,转成[0,1]相似度分数

        输入:
         - question:用户问题

        输出:
         - vector_scores_normalized: numpy数组,每个元素是语料中对应文档的[0,1]相似度

        用了什么模块:
         - self.vector_store.embed_query():问题向量化(新增方法)
         - self.vector_store.get_all_docs_and_vectors():读全量文档向量(新增方法)
         - np.linalg.norm(..., axis=1):按行算l2距离
        """
        # 问题向量化
        query_embedding = np.array(self.vector_store.embed_query(question))
        # 读全部文档文本和向量(注意:此处不重复调embedding API)
        documents, doc_embeddings = self.vector_store.get_all_docs_and_vectors()
        doc_embeddings = np.array(doc_embeddings)
        # 算l2距离:距离越小越相似
        distances = np.linalg.norm(query_embedding - doc_embeddings, axis=1)
        # 距离转相似度并归一化: 1 - (距离-最小距离)/(最大距离-最小距离)
        max_d = distances.max()
        min_d = distances.min()
        if max_d == min_d:
            # 极端情况,所有向量距离相同,返回全1
            return np.ones_like(distances)
        vector_scores_normalized = 1 - (distances - min_d) / (max_d - min_d)
        return vector_scores_normalized

    def search(self, question, top_k=TOP_K):
        "混合检索入口:融合BM25和向量分数,返回top_k个文档"
        """
        解决什么问题:
         - 拿到BM25归一化分数和向量归一化分数
         - 加权融合: combined = BM25_WEIGHT*bm25 + (1-BM25_WEIGHT)*vector
         - 按combined降序排序,取top_k个文档原文

        输入:
         - question:用户问题
         - top_k:返回的文档数,来自config

        输出:
         - hybrid_results: 融合排序后最相关的top_k个文档文本的列表

        用了什么模块:
         - self.bm25_retriever.search():BM25打分
         - self.vector_search_scores():向量打分
         - combined_scores.argsort()[::-1]:降序排序取索引
        """
        # BM25分数(归一化)
        bm25_scores = self.bm25_retriever.search(question)
        # 向量相似度分数(归一化)
        vector_scores = self.vector_search_scores(question)
        # 加权融合
        combined_scores = self.bm25_weight * bm25_scores + (1 - self.bm25_weight) * vector_scores
        # 降序排序,取top_k个索引
        top_index = combined_scores.argsort()[::-1][:top_k]
        # 按索引取回文档原文
        hybrid_results = [self.chunks[i] for i in top_index]
        return hybrid_results