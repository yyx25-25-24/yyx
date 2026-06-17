# 通用文档智能问答系统
基于RAG（检索增强生成）技术的通用文档问答系统，支持上传PDF、Word、TXT等多格式文档。系统采用TF-IDF向量检索与大模型结合的方式，提供精准、低幻觉的文档问答服务。

## 核心特性
- 支持PDF、Word(.docx)、TXT、Markdown多格式文档的解析与上传
- 内置独立检索模块，采用中文二元组分词、TF-IDF向量化与余弦相似度排序，精准定位文档相关片段
- 内置模型路由策略，自动识别问题复杂度。简单事实类问题通过本地检索快速返回；复杂推理类问题调用火山方舟大模型生成
- 双重缓存机制，包含精确匹配缓存与基于向量相似度的近似缓存，显著降低响应延迟与API调用成本
- 安全设计：API密钥通过`.env`环境变量管理，避免硬编码泄露；限制上传文件大小为16MB，降低恶意请求风险
- 自带效果测评模块，可从准确率、幻觉率、响应时延等维度量化评估系统性能

## 项目结构
```
问答/
├── .env                # API密钥配置文件（不提交到Git）
├── .gitignore          # Git忽略规则配置
├── requirements.txt    # Python依赖清单
├── README.md           # 项目说明文档
└── webapp/
    ├── app.py              # Flask主程序，负责路由与业务逻辑
    ├── retrieval.py        # 独立检索模块（分词、TF-IDF、相似度计算）
    ├── evaluation_system.py # 效果测评模块
    ├── templates/          # 前端HTML模板
    │   ├── layout.html     # 基础布局模板
    │   ├── upload.html     # 文件上传页面
    │   ├── files.html      # 文件列表与管理页面
    │   ├── qa.html         # 智能问答交互页面
    │   └── preview.html    # 文档预览页面
    └── uploads/            # 用户上传文件存储目录
```

## 环境依赖
```bash
pip install flask python-docx PyPDF2 python-dotenv volcenginesdkarkruntime
```
也可直接通过依赖文件批量安装：
```bash
cd 问答
pip install -r requirements.txt
```

## 快速开始
1.  安装依赖
```bash
cd 问答
pip install -r requirements.txt
```

2.  配置API密钥
在项目根目录创建`.env`文件，填入火山方舟API密钥：
```
ARK_API_KEY=你的火山方舟API密钥
```

3.  启动服务
```bash
cd webapp
python app.py
```

4.  访问系统
浏览器打开地址：http://localhost:5000

## 配置说明
主要参数在 `webapp/app.py` 顶部定义：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| TOP_K | 3 | 检索返回的相关片段数量 |
| MAX_CACHE_SIZE | 100 | 问答缓存最大条目数 |
| SIMILARITY_THRESHOLD | 0.9 | 相似问题判定阈值（用于缓存复用） |
| TEMPERATURE | 0.01 | 模型温度，数值越低输出越稳定 |
| MAX_CONTENT_LENGTH | 16MB | 允许上传的最大文件大小 |
| HOST | 127.0.0.1 | 监听地址，云部署改为0.0.0.0 |
| PORT | 5000 | Web服务端口 |

## 效果测评
运行测评脚本，可从准确率、幻觉率、响应时延、调用成本多个维度量化评估系统性能：
```bash
cd webapp
python evaluation_system.py
```
测评结果可输出为文本文件，用于对比不同版本的迭代效果。

## 云服务器部署
### 1. 环境准备
推荐使用Ubuntu 22.04系统，连接服务器后安装基础环境：
```bash
sudo apt update && sudo apt install python3 python3-pip -y
pip3 install flask python-docx PyPDF2 python-dotenv volcenginesdkarkruntime
```

### 2. 上传代码
在本地使用scp命令上传整个项目目录：
```bash
scp -r D:/练习代码/问答 ubuntu@你的服务器IP:/home/ubuntu/
```

### 3. 配置与启动
在服务器上创建`.env`文件并填入API密钥，进入webapp目录后台启动服务：
```bash
cd ~/问答/webapp
nohup python3 app.py > output.log 2>&1 &
```

### 4. 端口开放
在云服务器控制台的安全组中，开放TCP协议的5000端口，即可通过公网IP访问系统。

## 日志
服务运行后自动生成运行日志，记录用户提问、缓存命中、回答生成与异常报错信息，便于调试排查与效果分析。