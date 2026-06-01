import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# 1. 数据文件配置：替换成你的Excel文件路径即可
# 支持单个Excel文件，或多个Excel文件列表
DATA_FILES = [
    os.path.join(PROJECT_ROOT, "data", "17县原始数据.xlsx"),
    os.path.join(PROJECT_ROOT, "data", "17县归一化数据.xlsx"),
    os.path.join(PROJECT_ROOT, "data", "17县-SBM代码运行数据.xlsx")
]

# 2. 大模型配置
# 通义千问API Key 建议使用环境变量 .env 或系统环境变量配置，避免硬编码到代码中
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0))  # 0=最准确，无随机发挥
# 额外后端配置（可选）
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")  # auto|tongyi|openai|local
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", "2"))
REQUEST_BACKOFF = float(os.getenv("REQUEST_BACKOFF", "1.0"))

# 3. 可视化配置
# 生成的图表保存路径
CHART_SAVE_PATH = os.path.join(PROJECT_ROOT, "charts")
# 图表格式：支持 html/png/jpg，建议默认使用 html
CHART_FORMAT = os.getenv("CHART_FORMAT", "html").lower()

# 4. 分析报告配置
# 报告保存路径
REPORT_SAVE_PATH = os.path.join(PROJECT_ROOT, "reports")
# 报告格式：支持 docx/md
REPORT_FORMAT = os.getenv("REPORT_FORMAT", "md").lower()

# 5. 日志配置
LOG_SAVE_PATH = os.path.join(PROJECT_ROOT, "logs")

# 持久化会话/历史（简单 JSON 存储）
SESSION_STORE = os.path.join(PROJECT_ROOT, "session_history.json")
# 请求限流与成本日志
RATE_LIMIT_PER_SECOND = float(os.getenv("RATE_LIMIT_PER_SECOND", "5.0"))
COST_LOG_PATH = os.path.join(PROJECT_ROOT, "llm_costs.log")
