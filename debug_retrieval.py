# debug_retrieval.py
# 混合检索诊断脚本:定位"周鸿祎怎么说deepseek"为什么检索不到
#
# 排查顺序:
#  1. 语料里到底有没有"周鸿祎"(原文层面)
#  2. 问题分词与语料分词是否一致(分词层面)
#  3. 向量库与语料是否对齐(融合层面)
#  4. 含"周鸿祎"的chunk最终排第几,top5是哪些

import jieba
import numpy as np

import document_loader_ext
import vector_store
import hybrid_search
from config import TOP_K, BM25_WEIGHT

QUESTION = "周鸿祎怎么说deepseek"


def main():
    # ---------- 1. 复现ingest,拿到语料 ----------
    loader = document_loader_ext.DocumentLoaderExt()
    chunks = loader.splitter(loader.file_read())
    print(f"[1] 语料chunk总数: {len(chunks)}")

    # ---------- 2. 原文层面:关键词在不在语料里 ----------
    print("\n[2] 原文关键词命中情况:")
    hit_zhou = []
    for kw in ["周鸿祎", "周鸿", "deepseek", "DeepSeek"]:
        hit = [i for i, c in enumerate(chunks) if kw in c]
        if kw in ("周鸿祎", "周鸿"):
            hit_zhou = hit
        print(f"    {kw!r}: 命中 {len(hit)} 个chunk, 下标 {hit[:10]}")
    if not hit_zhou:
        print("    !! 语料里没有'周鸿祎':检索不可能命中,先查PDF是否包含该内容/解析是否正常")

    # ---------- 3. 分词层面:问题与语料分词是否一致 ----------
    print("\n[3] 分词对比:")
    print("    问题分词:", jieba.lcut(QUESTION))
    for i in hit_zhou[:3]:
        print(f"    chunk[{i}] 分词:", jieba.lcut(chunks[i]))
        print(f"    chunk[{i}] 原文:", chunks[i][:100])

    # ---------- 4. 对齐检查 + 三类打分 ----------
    store = vector_store.VectorStore()
    retriever = hybrid_search.HybridRetriever(chunks, store)

    store_docs, _ = store.get_all_docs_and_vectors()
    same = len(store_docs) == len(chunks) and all(a == b for a, b in zip(store_docs, chunks))
    print(f"\n[4] 向量库{len(store_docs)}条, 语料{len(chunks)}条, 顺序一致: {same}")

    bm25_scores = retriever.bm25_retriever.search(QUESTION)
    vec_scores = retriever.vector_search_scores(QUESTION)
    combined = BM25_WEIGHT * bm25_scores + (1 - BM25_WEIGHT) * vec_scores

    # ---------- 5. 含"周鸿祎"的chunk排名 ----------
    if hit_zhou:
        print("\n[5] 含'周鸿祎'的chunk分数与排名:")
        for i in hit_zhou:
            rank = int((combined > combined[i]).sum()) + 1
            print(f"    chunk[{i}] bm25={bm25_scores[i]:.3f} 向量={vec_scores[i]:.3f} "
                  f"融合={combined[i]:.3f} -> 排第{rank}名")

    # ---------- 6. 最终top5 ----------
    top_idx = combined.argsort()[::-1][:TOP_K]
    print(f"\n[6] 融合检索最终top{TOP_K}: {list(top_idx)}")
    for i in top_idx:
        print(f"    --- chunk[{i}] bm25={bm25_scores[i]:.3f} 向量={vec_scores[i]:.3f} 融合={combined[i]:.3f}")
        print(f"        {chunks[i][:80]}")


if __name__ == "__main__":
    main()