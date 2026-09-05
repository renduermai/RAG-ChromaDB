from document_loader import DocumentLoader
# from embedding_client import EmbeddingClient
from llm_client import LLMClient
from models import get_ali_embeddings
from vector_store import VectorStore
from langchain_chroma import Chroma
from config import CHROMA_PATH, COLLECTION_NAME, TOP_K, BATCH_SIZE



class RAGPipeline:

    # 改前代码
    # def __init__(self):
    #     self.document_loader = DocumentLoader()
    #     self.embedding_client = EmbeddingClient()
    #     self.vector_store = VectorStore()
    #     self.llm_client = LLMClient()

    # 改后代码
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.vector_store = VectorStore()
        self.llm_client = LLMClient()

    # 改前代码
    # def ingest(self,file_path):
    #     # 文件读取
    #     file_contents = self.document_loader.load(file_path)
    #     # 切分文档
    #     results = self.document_loader.text_splitter(file_contents)
    #     # 向量化
    #     vector_chunks = self.embedding_client.embed_documents(results)
    #     # 存储
    #     self.vector_store.add_documents(results,vector_chunks)

    # 改后代码
    def ingest(self, file_path):
        # 文件读取
        file_contents = self.document_loader.load(file_path)
        # 切分文档
        results = self.document_loader.text_splitter(file_contents)
        # 存储（向量化由向量库内部自动完成）
        self.vector_store.add_documents(results)

    # 改前代码
    # def query(self,query):
    #     # 问题向量化
    #     vector_query = self.embedding_client.embed_query(query)
    #     # 搜索 top_k 文档
    #     top_k_docs = self.vector_store.search(vector_query)
    #     documents = top_k_docs.get("documents",[[]])[0]
    #     # 拼接文档
    #     context = "\n".join(documents)
    #     # 问题和文档给 AI 返回答案
    #     answer = self.llm_client.rag_ask(query,context)
    #     return answer

    # 改后代码
    def query(self, query):
        # 搜索 top_k 文档
        top_k_docs = self.vector_store.search(query)

        # 拼接文档
        context = "\n".join(top_k_docs)
        # 问题和文档给 AI 返回答案
        answer = self.llm_client.rag_ask(query,context)
        return answer

    def chat(self):
        print("欢迎进入 AI-RAG 问答系统，按 q 退出")

        while True:
            # 接收问题
            question = input("请输入问题：")
            if question.lower() in ['q','quit','exit']:
                print('再见')
                break
            if not question:
                continue
            try:
                answer = self.query(question)
                print(f'ai回答：{answer}')
            except Exception as e:
                print(e)


        # 循环，退出
