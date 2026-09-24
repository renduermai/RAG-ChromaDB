from RagPipeline import RagPipeline
from config import DATA_PATH

def main():
    rag_pipeline = RagPipeline()
    rag_pipeline.ingest()
    rag_pipeline.chat()



if __name__ == "__main__":
    main()