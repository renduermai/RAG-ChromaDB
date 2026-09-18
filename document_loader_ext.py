# document_loader_ext.py
# 多格式文档加载扩展模块
#
# 解决什么问题:
#  - 原 document_loader 只支持 .txt,本模块在不改动原代码的前提下
#    新增 .pdf / .doc / .docx / .jpeg / .jpg 等格式识别

# 核心思路:
#  - 建一张"后缀名 → 加载器"映射表(LOADER_TABLE)
#  - 继承原 DocumentLoader,只重写 file_read() 一个方法:
#    识别后缀 → 图片走 image_reader(大模型OCR)
#             → 命中映射表走对应加载器
#             → 否则调用父类(原txt逻辑)
#  - splitter() 切块函数完全继承父类,零改动


import os
import re
import unicodedata

from config import DATA_PATH
from document_loader import DocumentLoader

# 【新增】图片OCR模块:用视觉大模型提取图片文字,不装任何本地软件
import image_reader

# pdf 加载器:逐页读取pdf文本
from langchain_community.document_loaders import PyPDFLoader

# word 加载器:读取 .docx / .doc 文本
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

# ======================文本净化(新增:解决PDF提取出"假汉字"的问题)======================
# 解决什么问题:
#  - 百度百科等网页导出的PDF,提取出的"示/月/人/网/方/面/支/力/十/大/高/金"等字
#    实际是"康熙部首"兼容字符(码点U+2F00~U+2FDF,肉眼几乎一样),
#    例如"表⽰"的⽰是U+2F70,不是普通的"示"(U+793A)
#    后果:jieba分词错乱("周鸿祎表⽰"被切成"周鸿祎表"+"⽰"),BM25词面对不上导致漏检,
#    向量化和喂给大模型的文本也带脏字符
#  - 文本里还夹着 \x01 等控制字符
# 做法:
#  - NFKC规范化:兼容字符还原为标准汉字(⽰->示,⽉->月,⼈->人),顺带全角转半角
#  - 正则去掉控制字符,保留 \n \t \r(切块还要靠换行)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text):
    "净化加载器提取的文本:兼容字符还原 + 去控制字符"
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    return text

# ======================图片格式清单(新增图片格式=改这里)======================
# 解决什么问题:
#  - 单独列出图片类后缀,因为图片不走映射表,而是走大模型OCR
IMAGE_EXTS = [".jpeg", ".jpg",".png"]

# ======================格式映射表(新增非图片格式=新增一行)======================
# 解决什么问题:
#  - 用一张表集中管理"什么后缀用什么加载器",避免写一堆if/elif
# 输入:文件后缀名(小写)   输出:对应的langchain加载器类
LOADER_TABLE = {
    ".pdf":  PyPDFLoader,           # pdf 文档
    ".doc":  Docx2txtLoader,         # word 老格式
    ".docx": Docx2txtLoader,
    ".md": TextLoader  # markdown 格式

}


class DocumentLoaderExt(DocumentLoader):
    "多格式文档加载器:识别文件后缀,自动选择对应的读取方式"

    def __init__(self, file_path=DATA_PATH):
        "初始化:直接复用父类,不需要额外准备任何工具"
        """
        解决什么问题:
         - 获得文件地址(与原模块完全一致,只是转了一道父类)

        输入:
         - file_path:文件地址,来自config

        输出:
         - self.file_path

        用了什么模块:
         - super():调用父类DocumentLoader的初始化,一行顶三行
        """

        super().__init__(file_path)
        # 新增:初始化图片OCR工具,它内部会自动连接阿里百炼视觉大模型
        self.image_reader = image_reader.ImageReader()

    def file_read(self):
        "按后缀名识别格式,读取文件内容"
        """
        解决什么问题:
         - 识别文件是什么格式(看后缀名)
         - 命中映射表:用对应的langchain加载器读取
         - 未命中:回退到父类的txt读取逻辑,保持原功能不变

        输入:
         - self.file_path:文件地址

        输出:
         - 读取内容的纯文本字符串(与父类输出格式一致,后续流程无感)

        用了什么模块:
         - os.path.splitext():拆出文件后缀名,例如 "a.pdf" -> ".pdf"
         - LOADER_TABLE:后缀名->加载器的映射表
         - load():langchain各加载器的读取方法,返回Document列表
         - doc.page_content:langchain读取文本的key
         - super().file_read():父类的txt读取,兜底
        """
        # 拆出后缀名并转小写,统一大小写,例如 ".PDF" ".pdf" 都能识别
        ext = os.path.splitext(self.file_path)[-1].lower()

        # 分支1:图片格式 -> 走视觉大模型OCR(新增的唯一分支)
        if ext in IMAGE_EXTS:
            # return self.image_reader.read(self.file_path)
            return clean_text(self.image_reader.read(self.file_path))
        # 分支2:命中映射表 -> pdf/word 走对应的langchain加载器
        if ext in LOADER_TABLE:
            documents = LOADER_TABLE[ext](self.file_path).load()
            # 有的格式读取结果是多页/多段,合并成一整段文本字符串
            # return "\n".join(doc.page_content for doc in documents)
            return clean_text("\n".join(doc.page_content for doc in documents))
        # 分支3:其余格式(如txt)走父类原有逻辑,一行不改
        return super().file_read()

    # 注意:splitter() 切块函数完全继承父类,无需重写