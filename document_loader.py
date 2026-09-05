# 文档加载、切分分块
from langchain_community.document_loaders import TextLoader

from config import CHUNK_SIZE,CHUNK_OVERLAP,SEPARATORS
from langchain_text_splitters import RecursiveCharacterTextSplitter
class DocumentLoader:

    # @staticmethod
    # def load(file_path):
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         contents = f.read()
    #         return contents

    # 文件读取（langchain）
    @staticmethod
    def load(file_path):
        contents = TextLoader(file_path, encoding="utf-8").load()
        return contents[0].page_content

    @staticmethod
    def text_splitter(contents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
            )

        chunks = splitter.split_text(contents)
        return chunks