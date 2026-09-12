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
