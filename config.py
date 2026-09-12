import os
from dotenv import load_dotenv

# 加载 .env文件
load_dotenv()

# ======================文件存储路径======================
# 数据文件路径
DATA_PATH = os.path.join(os.path.dirname(__file__),"DATA","deepseek百度百科.txt")

# chromaDB持久化路径
CHROMA_PATH =os.path.join(os.path.dirname(__file__),"DATA","CHROMA_DATA")

# ======================模型配置 API key==================
# 阿里云
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY","")
DASHSCOPE_API_URL = os.getenv("DASHSCOPE_API_URL","")
DASHSCOPE_API_MODEL = os.getenv("DASHSCOPE_API_MODEL","")
DASHSCOPE_API_EMBEDDING_MODEL = os.getenv("DASHSCOPE_API_EMBEDDING_MODEL","")

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY","")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL","")
DEEPSEEK_API_MODEL = os.getenv("DEEPSEEK_API_MODEL","")

# ====================== 文本切分配置 ======================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE","150"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP","30"))
SEPARATORS = ["\n\n","\n","。","？","，",""] # 分隔符优先级

# ====================== 批处理设置 ======================
BATCH_SIZE = int(os.getenv("BATCH_SIZE","10")) # 防止 API 限流
TOP_K = int(os.getenv("TOP_K","5"))
TEMPERATURE = float(os.getenv("TEMPERATURE","0")) # 温度

# ====================== 集合名称 ======================
COLLECTION_NAME = "demo"