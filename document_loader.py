from config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    "完成文档加载,切分,输出切块后的文档"

    def __init__(self, file_path=DATA_PATH):
        "初始化 读,切块函数需要的文件"
        """
        解决什么问题:
         - 初始化文件

        输入:
         - file_path:文件地址,来自config

        输出:
         - self.file_path

        用了什么模块:
         DATA_PATH(from config import DATA_PATH):config里面的文件地址

        """
        self.file_path = file_path

    def file_read(self):
        "读取文件,加载到数据库,但不向量化"
        """
        解决什么问题:
         - 文件读取
         - 文件加载
         - 返回读取的内容

        输入:
         - self.file_path

        输出:
         - 读取内容的文本内容

        用了什么模块:
         - self.flie_path: 文件地址
         - TextLoader:from langchain_community.document_loaders import TextLoader
         - load():文档加载,TextLoader的方法
         - page_content:langchian读取文本的key
        """
        document = TextLoader(self.file_path, encoding='utf-8').load()
        return document[0].page_content

    def splitter(self, document):
        "把加载的文档用langchian切块"
        """
        解决什么问题:
         - 递归切块文本
         - 返回切块后的文本

        输入:
         - document:上一步加载的文本

        输出:
         - chunk:切块的文档

        用了什么模块:
         - RecursiveCharacterTextSplitter:递归切块规则.from langchain_text_splitters import RecursiveCharacterTextSplitter
         - split_text():langchain的切分文档函数
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
        )

        chunk = splitter.split_text(document)
        return chunk