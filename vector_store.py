from langchain_chroma import Chroma
from config import COLLECTION_NAME, CHROMA_PATH, BATCH_SIZE, TOP_K
from models import get_ali_embeddings


class VectorStore:
    "把切块后的文档向量化,存储;以及可以根据问题检索top_k文档"

    def __init__(self, persist_path=CHROMA_PATH):
        "初始化向量数据库模块"
        """
        解决什么问题:
         - 初始化Chroma模块:相当于新建表
         - 接通向量化api
         - 确认本地存储地址

        输入:
         - CHROMA_PATH: 来自config

        输出:
         - self.store:初始化的Chroma模块

        用了什么模块:
         - config:存储地址
         - Chroma模块:from langchain_chroma import chroma
         - get_ali_embedings:model模块导入向量化模型

        """
        self.store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=persist_path,
            embedding_function=get_ali_embeddings(),
            collection_metadata={"hnsw:space": "l2"}
        )

    def add_document(self, chunk, ids=None, batch_size=BATCH_SIZE):
        "把切分好的文档存到数据库"
        """
        解决了什么问题:
         - 把文档和id匹配
         - 分批次存储

        输入:
         - chunk:切块的文档
         - ids:文档id
         - BATCH_SIZE:批次大小


        输出:
         - 无(存好即可)

        用了什么模块:
         - add_texts():Chroma模块用于向量化+存储的函数
         - BATCH_SIZE:config模块

        """
        if ids is None:
            ids = [str(i) for i in range(len(chunk))]
        for i in range(0, len(chunk), batch_size):
            self.store.add_texts(
                ids=ids[i:i + batch_size],
                texts=chunk[i:i + batch_size]
            )

    def search(self, question, top_k=TOP_K):
        "把问题向量化,并检索top_k个相关文档"
        """
        解决什么问题:
         - 问题向量化+检索(langchian模块)
         - 返回检索的文档文本

        输入:
         - question:问题
         - top_k:config里面的TOP_K

        输出:
         - doc:和问题相关的top_k个向量文档的内容

        用了什么模块:
         - self.store:调用初始化的Chroma模块
         - similarity_search:Chroma模块相似度检索模块,包含向量化+检索    
         - config:初始化top_k

        """

        docs = self.store.similarity_search(
            query=question,
            k=top_k
        )
        doc = [doc.page_content for doc in docs]
        return doc


# ====================== 新增:混合检索需要的两个方法 ======================

    def embed_query(self, question):
        "把问题单独向量化(不检索)"
        """
        解决什么问题:
         - 混合检索需要对问题单独算一个向量,用于和全量文档向量算距离
         - 不再重复调检索接口,只取向量化结果

        输入:
         - question:用户问题

        输出:
         - query_embedding:问题的向量(list)

        用了什么模块:
         - self.store._embedding_function:Chroma初始化时传入的向量化模型,直接复用
         - embed_query():langchain向量化模型的向量化单个文本的函数
        """
        return self.store._embedding_function.embed_query(question)

    def get_all_docs_and_vectors(self):
        "读出向量数据库中的全部文档文本和向量"
        """
        解决什么问题:
         - 混合检索需要对全量语料打分(BM25和向量各打一遍),而不是只取top_k
         - 向量已经在入库时算好,直接从库里读,不重复调embedding API

        输入:
         - 无

        输出:
         - documents:全部文档文本的列表(list)
         - embeddings:全部文档向量的列表(list)

        用了什么模块:
         - self.store.get():Chroma模块,按条件取出collection中的数据
         - include:指定同时返回文档内容(documents)和向量(embeddings)

        注意:
         - 文档顺序与ingest时切块顺序一致(add_document默认按0,1,2...生成id顺序入库)
        """
        result = self.store.get(include=["documents", "embeddings"])
        return result["documents"], result["embeddings"]