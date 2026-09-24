# image_reader.py
# 视觉大模型OCR模块
#
# 解决什么问题:
#  - jpeg/jpg 图片里的文字,不装任何本地软件(tesseract等)
#  - 直接调用阿里云百炼的视觉大模型(qwen系列)完成图片文字提取
#
# 核心思路:
#  - 读图片文件 -> 转成base64 -> 按OpenAI视觉格式发给大模型
#  - 大模型返回图片中的文字 -> 交给原有切分流程

import base64

from config import DASHSCOPE_API_VL_MODEL
from models import get_ali_vl_model_client


class ImageReader:
    "用视觉大模型,把图片里的文字读出来"

    def __init__(self, model=DASHSCOPE_API_VL_MODEL):
        "初始化:接通视觉大模型"
        """
        解决什么问题:
         - 获得视觉大模型的客户端

        输入:
         - model:视觉大模型名,来自config(默认qwen-vl-max)

        输出:
         - self.model:模型名
         - self.client:大模型客户端(和文本大模型同一个平台,同一个key)

        用了什么模块:
         - get_ali_vl_model_client:models模块已有函数,直接复用,一行不改
         - config:DASHSCOPE_API_VL_MODEL视觉模型名
        """
        self.model = model
        # 复用models.py里现成的客户端函数,不传参数就是默认的阿里百炼平台
        self.client = get_ali_vl_model_client()

    def read(self, file_path):
        "读取图片文件,返回图片中的文字"
        """
        解决什么问题:
         - 图片文件 -> base64编码
         - 按OpenAI视觉格式组装消息(文字指令 + 图片)
         - 调用视觉大模型,提取图片中的全部文字

        输入:
         - file_path:图片文件地址(例如 a.jpeg)

        输出:
         - text:图片中提取出的文字(纯文本字符串,和txt读取结果格式一致)

        用了什么模块:
         - base64:python内置,文件转base64
         - self.client.chat.completions.create:OpenAI格式的视觉对话接口
         - data:image/jpeg;base64:OpenAI视觉格式的图片传参方式
         - .choices[0].message.content:提取大模型回答内容
        """
        # 1.读图片二进制 -> base64编码(网络传输图片的标准方式)
        with open(file_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 2.组装视觉消息:一条文字指令 + 一张base64图片
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "请完整提取这张图片中的所有文字,"
                                         "按原有排版输出,不要添加任何解释。"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ]
        }]

        # 3.调用视觉大模型
        response = self.client.invoke(messages)

        # 4.取回答文字
        text = response.content
        return text