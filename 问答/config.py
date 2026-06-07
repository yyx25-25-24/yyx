# -------------------------- 所有配置都在这里，以后改东西只需要改这个文件 --------------------------
# 论文文件路径
import os
# 从系统环境变量读取API密钥，避免明文写在代码里
ARK_API_KEY = os.getenv("ARK_API_KEY", "请设置环境变量ARK_API_KEY")
DOCUMENT_PATH = "D:/练习代码/修改2.docx"

# 火山方舟API配置
ARK_API_KEY = "ark-72069a38-1275-4c3a-96b9-98542a340de0-79e39"
MODEL_NAME = "doubao-seed-1-8-251228"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# Web服务配置
HOST = "127.0.0.1"  # 本地开发用 127.0.0.1，云服务器部署改为 "0.0.0.0"
PORT = 7860
LOG_FILE = "rag.log"  # 日志文件路径（相对项目目录）

# RAG参数配置
CHUNK_SIZE = 800  # 每个文本块的字数
TOP_K = 3  # 检索最相关的3个片段
TEMPERATURE = 0.01  # 模型温度，越低越准确
MAX_TOKENS = 1024  # 回答的最大长度

# 缓存配置
MAX_CACHE_SIZE = 100  # 最大缓存条数，避免内存无限增长
