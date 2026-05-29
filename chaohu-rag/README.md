# 巢湖流域污染分析智能问答系统

基于 RAG（检索增强生成）技术的学术论文问答系统，所有回答均来自论文原文，杜绝幻觉。

## 项目结构

```
问答/
├── config.py           # 所有配置集中在这里
├── document.py         # 文档加载和智能分块
├── retrieval.py        # TF-IDF 向量检索引擎
├── server.py           # Web 服务器和主程序
├── evaluation.py       # BM25 vs TF-IDF 对比实验
├── rag.log             # 运行日志（自动生成）
├── evaluation_results.txt  # 对比实验结果（运行 evaluation.py 后生成）
└── README.md           # 项目说明文档
```

## 环境依赖

```bash
pip install python-docx volcenginesdkarkruntime
```

## 快速开始

1. 在 `config.py` 中配置论文路径 `DOCUMENT_PATH` 和火山方舟 `ARK_API_KEY`
2. 启动服务：

```bash
cd 问答
python server.py
```

3. 浏览器访问：http://localhost:7860

## 配置说明

所有可调参数均在 `config.py`：

| 参数 | 说明 |
|------|------|
| `DOCUMENT_PATH` | Word 论文文件路径 |
| `ARK_API_KEY` | 火山方舟 API Key |
| `CHUNK_SIZE` | 文本分块大小（字符数） |
| `TOP_K` | 检索返回的相关片段数量 |
| `TEMPERATURE` | 模型温度，越低越稳定 |
| `HOST` | 监听地址，本地 `127.0.0.1`，云部署改为 `0.0.0.0` |
| `PORT` | Web 服务端口 |
| `MAX_CACHE_SIZE` | 问答缓存上限（相同问题秒回） |
| `LOG_FILE` | 日志文件名 |

## 运行对比实验

```bash
python evaluation.py
```

会对比 BM25 基线与 TF-IDF 在检索速度、Top-1 相似度、关键词覆盖率上的差异，结果保存到 `evaluation_results.txt`，可直接用于论文。

## 云服务器部署

### 1. 申请服务器

- 阿里云免费试用：https://free.aliyun.com/
- 腾讯云轻量服务器：https://cloud.tencent.com/act/free

推荐 Ubuntu 22.04 系统。

### 2. 连接并安装环境

```bash
ssh ubuntu@你的服务器公网IP

sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip unzip -y
pip3 install python-docx volcenginesdkarkruntime
```

### 3. 上传代码

在本地打包上传：

```bash
# 本地（PowerShell）
scp -r D:/练习代码/问答 ubuntu@你的服务器IP:/home/ubuntu/
```

同时上传论文文件，并修改服务器上 `config.py` 的 `DOCUMENT_PATH` 和 `ARK_API_KEY`。

### 4. 修改配置并启动

编辑 `config.py`：

```python
HOST = "0.0.0.0"   # 允许外网访问
DOCUMENT_PATH = "/home/ubuntu/问答/修改2.docx"  # 改为服务器上的路径
```

后台运行：

```bash
cd ~/问答
nohup python3 server.py > output.log 2>&1 &
```

### 5. 开放端口

在云控制台安全组中，添加入站规则：TCP 端口 `7860`。

### 6. 访问

浏览器打开：http://你的服务器公网IP:7860

## 日志

运行后自动生成 `rag.log`，记录用户提问、缓存命中、新回答生成和错误信息，便于调试和答辩展示。
