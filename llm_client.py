
from config import TEMPERATURE
from models import get_ali_model_client

# 新增
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# 旧代码
# class LLMClient:
#     def __init__(self):
#         self.client = get_ali_model_client(
#             temperature=TEMPERATURE
#         )
#
#     def ask(self,query):
#         responses = self.client.invoke(query)
#         content = responses.content if hasattr(responses, "content") else str(responses)
#         return content
#
#     def rag_ask(self,query, context):
#         prompt = self._build_rag_prompt(query, context)
#         return self.ask(prompt)
#
#     @staticmethod
#     def _build_rag_prompt(query, context):
#         return f"""
#         你是一个问答机器人。
#         你的任务是更具下述给定的已知信息回答用户问题。
#         确保你的回复完全依据下述已知信息。不要编造答案。
#         你需要判断，如果下述已知信息不足以回答用户问题，请直接回复“信息不足，我无法回答您的问题”
#
#         问题：{query}
#         已知信息：{context}
#
#         用中文回答
#         """

# 新代码
RAG_PROMPT = ChatPromptTemplate.from_template("""
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
你需要判断，如果下述已知信息不足以回答用户问题，请直接回复"信息不足，我无法回答您的问题"

问题：{query}
已知信息：{context}

用中文回答
""")

class LLMClient:
    def __init__(self):
        self.llm = get_ali_model_client(temperature=TEMPERATURE)
        # LCEL 链
        self.chain = RAG_PROMPT | self.llm | StrOutputParser()

    def ask(self, query):
        return self.llm.invoke(query).content

    def rag_ask(self, query, context):
        return self.chain.invoke({"query": query, "context": context})