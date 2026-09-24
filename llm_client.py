from config import TEMPERATURE
from models import get_ali_model_client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

RAGPrompt = ChatPromptTemplate.from_template(
    """
    你是一个问答助手,根据用户问过检索答案
    要求来源准确,不得随意编造
    如果为检索到,就回答:对不起,未检索到相关内容
    ---
    已知信息:{context}
    问题:{question}
    """
)


class LLMClient():
    "把问题+检索的相关文档给AI,结合提示词,输出回答"
    """
    调用大模型回答问题的类

    在pipeline位置:检索问题相关文档之后

    解决了什么问题:
     - 回答问题
     - 根据上下文回答问题
     - 解析回答

    用了什么模块:
     - ChatPromptTemplate(langchain_core):结构化提示词
     - StrOutputParser:解析回答
     - LECL:链式调用

    """

    def __init__(self, temperature=TEMPERATURE):
        "初始化大模型和思维链"
        """
        解决什么问题:
         - 初始化大模型
         - 初始化LECL链

        输入:
         - temperature:温度,来自config

        输出:
         - self.llm:ai直接回答
         - self.chain:链式回答

        用了什么模块:
         - get_ali_model_client:models模块;from models import get_ali_model_client
         - config:温度
         - StrOutputParser():langchain 规范输出格式
        """
        self.llm = get_ali_model_client(temperature=temperature)
        self.chain = RAGPrompt | self.llm | StrOutputParser()

    def ask(self, question):
        """
        单次大模型问答

        解决什么问题:
         - 调用llm
         - 提取回答内容

        输入:
         - question:问题

        输出:
         - answer:大模型回答内容

        用到的模块:
         - self.llm.invoke():调用大模型
         - .content:提取回答内容
        """
        answer = self.llm.invoke(question).content
        return answer

    def rag_ask(self, context, question):  # 注意顺序和prompt一致
        """
        链式调用大模型回答Rag类型的问题

        解决了什么问题:
         - 把上下文作为背景context给大模型
         - 避免幻觉
         - 结构化处理输出

        输入:
         - query:问题
         - contexts:上一步检索到的相关topk文档

        输出:
         - answer:大模型回答的内容

        用了什么模块:
         - self.chain.invoke():链式调用
         - LCEL:链式调用:提示词--大模型--回答解析
        """
        answer = self.chain.invoke({ "question": question,"context": context})
        return answer
