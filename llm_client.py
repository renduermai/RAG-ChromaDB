from oauthlib.uri_validate import query

from config import TEMPERATURE
from models import get_ali_model_client


class LLMClient:
    def __init__(self):
        self.client = get_ali_model_client(
            temperature=TEMPERATURE
        )

    def ask(self,query):
        responses = self.client.invoke(query)
        content = responses.content if hasattr(responses, "content") else str(responses)
        return content

    def rag_ask(self,query, context):
        prompt = self._build_rag_prompt(query, context)
        return self.ask(prompt)

    @staticmethod
    def _build_rag_prompt(query, context):
        return f"""
        你是一个问答机器人。
        你的任务是更具下述给定的已知信息回答用户问题。
        确保你的回复完全依据下述已知信息。不要编造答案。
        你需要判断，如果下述已知信息不足以回答用户问题，请直接回复“信息不足，我无法回答您的问题”
        
        问题：{query}
        已知信息：{context}
        
        用中文回答
        """
