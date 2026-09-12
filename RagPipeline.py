# 目标(客户视角)
# 接受文件--向量化,存储,检索--提问回答--多次问,退出
import document_loader
import llm_client
import vector_store


class RagPipeline:
    "从文档读取-向量化-存储-问题检索-ai问答 的全流程管线"


    def __init__(self):
        "备齐管线其他函数所需要的工具"
        self.document_loader = document_loader.DocumentLoader()
        self.vector_store = vector_store.VectorStore()
        self.llm = llm_client.LLMClient()


    def ingest(self):
        "摄入文档"
        """
        解决什么问题:
         - 读取文件
         - 文件切块
         - 切块文档向量化并存储
    
        输入:
         - 无(文件在子模块自动加载,保持管线清爽)
    
        输出:
         - doc_vector:向量化后的文档(存到数据库即可)
    
        用到什么模块:
         - document_loader.file_read():文档加载类,读文件函数
         - document_loader.splitter():文档加载类,切分文档函数
         - vector_store.add_document():向量数据库类,存储文档
        """
        # 读取文件
        document = self.document_loader.file_read()
        # 切块
        chunk = self.document_loader.splitter(document)
        # 向量化+存储
        doc_vector = self.vector_store.add_document(chunk)

        return doc_vector


    def query(self, question):
        "一次AI检索问答"
        """
        解决什么问题:
         - 问题向量化,并在向量数据库中找到topk个相关文档
         - topk个向量文档拼接
         - 接通AI模块,输出基于文档的回答
    
        输入:
         - question:用户问题
    
        输出:
         - answer:基于问题+文档 的回答
    
        用了什么模块:
         - vector_store.search():向量数据库类,把问题向量化并检索topk文档的函数
         - llm.rag_ask():大模型类,接通AI,根据问题和文档回答的函数
    
        """
        # 问题向量化,检索topk个文档
        top_k_doc = self.vector_store.search(question)

        # topk个文档拼接
        context = "\n".join(top_k_doc)

        # AI回答
        answer = self.llm.rag_ask(context, question)
        return answer

    def chat(self):
        "多次问答,退出"
        """
        解决什么问题:
         - 退出:用户输入q,exit,quit自动转小写,识别并退出break
         - 空内容处理:用循环continue
         - 循环问答:While True
    
        输入:
         - 无(输入在子模块已处理)
    
        输出:
         - answer:调用query()函数,并print结果
    
        用到了什么方法:
         - query:循环调用回答
         - while循环:continue,break
    
        """
        print("你好,欢迎你进入AI-RAG问答系统~~")
        while True:
            question = input("请输入你的问题:")
            if question.lower() in ['q', 'quit', 'exit']:
                print("那就再见了")
                break

            if not question:
                continue

            answer = self.query(question)
            print(f"AI回答:{answer}")


