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

from config import DATA_PATH
from document_loader import DocumentLoader

# 【新增】图片OCR模块:用视觉大模型提取图片文字,不装任何本地软件
import image_reader

# pdf 加载器:逐页读取pdf文本
from langchain_community.document_loaders import PyPDFLoader

# word 加载器:读取 .docx / .doc 文本
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader


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
            return self.image_reader.read(self.file_path)

        # 分支2:命中映射表 -> pdf/word 走对应的langchain加载器
        if ext in LOADER_TABLE:
            documents = LOADER_TABLE[ext](self.file_path).load()
            # 有的格式读取结果是多页/多段,合并成一整段文本字符串
            return "\n".join(doc.page_content for doc in documents)

        # 分支3:其余格式(如txt)走父类原有逻辑,一行不改
        return super().file_read()

    # 注意:splitter() 切块函数完全继承父类,无需重写