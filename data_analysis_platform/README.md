# 通用型大模型自动化数据分析平台

这是一个基于 LangChain 和通义千问的大模型驱动自动化数据分析平台，支持任意结构化 Excel 数据的分析需求。用户只需通过自然语言输入分析指令，平台即可自动完成数据读取、分析、可视化图表生成和报告输出。

## 目录结构
```
data_analysis_platform/
├── config.py          # 通用配置文件
├── main.py            # 核心程序
├── requirements.txt   # 依赖清单
├── README.md          # 项目说明
├── .env.example       # 示例环境变量配置
├── data/              # 数据文件夹，存放Excel数据
│   ├── 17县原始数据.xlsx
│   ├── 17县归一化数据.xlsx
│   └── 17县-SBM代码运行数据.xlsx
├── charts/            # 自动生成的图表保存目录
├── reports/           # 自动生成的报告保存目录
└── logs/              # 分析交互日志保存目录
```

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 配置环境变量（可选）
如果你想启用通义千问大模型，请复制 `.env.example` 为 `.env`，并将 `DASHSCOPE_API_KEY` 填写为你的 API Key。若不配置该密钥，平台会自动回退到本地分析模式。

3. 将你的 Excel 数据文件放到 `data/` 文件夹，或修改 `config.py` 中的 `DATA_FILES` 路径。

4. 运行平台
```bash
python main.py
```

5. 在交互式命令行中输入分析需求，例如：
- 分析2012-2023年氮磷排放的总体变化趋势，生成折线图
- 找出2023年碳排放最高的5个区县，生成柱状图
- 分析数字化指数与氮磷排放的相关性，生成散点图

## 功能说明
- ✅ 通用数据适配：支持多个 Excel 文件、多个 sheet
- ✅ 自然语言分析：无须写代码，直接输入分析指令
- ✅ 自动可视化：自动生成图表并保存到 `charts/`
- ✅ 分析报告保存：将每次查询结果存储为 Markdown/Word 报告
- ✅ 日志记录：自动记录分析请求和结果，方便复现与审计

## 文件说明
- `config.py`：配置数据文件、模型参数、图表和报告保存目录
- `main.py`：平台入口，负责数据加载、Agent创建、命令交互和分析执行
- `requirements.txt`：项目依赖清单
- `README.md`：项目说明文档

## 注意
- 请将真实的 Excel 数据文件放入 `data/` 文件夹，或直接在 `config.py` 中修改 `DATA_FILES`
- 当前平台默认支持通义千问模型。若未配置 API Key，平台会自动回退到本地数据分析模式。
- 如果要替换为其他模型，请修改 `config.py` 中的 `MODEL_NAME` 和 `DASHSCOPE_API_KEY`。

## Web 原型（上传与导出报告）

本仓库包含一个轻量的 Flask 原型，用于通过网页上传数据、选择分析与图表参数，并导出包含图表的 HTML 报告。

- 启动命令：
```
python -m data_analysis_platform.webapp.app
```
- 配置：可以在 `.env` 中设置 `WEB_USERNAME` 和 `WEB_PASSWORD` 启用简单登录（留空则禁用）。
- 输出：生成的图表保存在 `charts/`，报告保存在 `reports/`（HTML，可选 PDF 需安装 `pdfkit` + `wkhtmltopdf`）。
- 会话历史：访问 `/history` 可以查看最近的分析请求、数据集和结果摘要。

前端在选择文件后会自动解析列并填充横轴/纵轴下拉，分析完成后会生成可在页面查看与下载的报告与图片。

## 测试

使用 `pytest` 运行 Web 原型的基本端到端测试：

```bash
pytest data_analysis_platform/tests/test_webapp_end2end.py data_analysis_platform/tests/test_webapp_history.py
```
