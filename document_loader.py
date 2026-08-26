# 文档加载、切分分块
from config import CHUNK_SIZE,CHUNK_OVERLAP,SEPARATORS
from langchain_text_splitters import RecursiveCharacterTextSplitter
class DocumentLoader:

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            contents = f.read()
            return contents

    @staticmethod
    def text_splitter(contents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
            )

        chunks = splitter.split_text(contents)
        return chunks