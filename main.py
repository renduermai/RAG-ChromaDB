from RAGPipeline import RAGPipeline
from config import DATA_PATH


def main():
    rag_pipeline = RAGPipeline()
    rag_pipeline.ingest(DATA_PATH)
    rag_pipeline.chat()

if __name__ == "__main__":
    main()