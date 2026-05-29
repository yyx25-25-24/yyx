import logging
from collections import OrderedDict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs

from volcenginesdkarkruntime import Ark

from config import (
    ARK_API_KEY,
    BASE_URL,
    HOST,
    LOG_FILE,
    MAX_CACHE_SIZE,
    MAX_TOKENS,
    MODEL_NAME,
    PORT,
    TEMPERATURE,
    TOP_K,
)
from document import load_and_split_document
from retrieval import TfidfRetriever

# -------------------------- 日志系统 --------------------------
LOG_PATH = Path(__file__).parent / LOG_FILE
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# -------------------------- 内存缓存（LRU） --------------------------
answer_cache = OrderedDict()


def _cache_get(question):
    if question not in answer_cache:
        return None
    answer_cache.move_to_end(question)
    return answer_cache[question]


def _cache_set(question, answer):
    answer_cache[question] = answer
    answer_cache.move_to_end(question)
    while len(answer_cache) > MAX_CACHE_SIZE:
        answer_cache.popitem(last=False)


def _format_retrieval_details(results):
    """组装检索详情，供前端展示"""
    details = "\n\n---\n### 检索详情\n"
    for i, (chunk, score) in enumerate(results):
        preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
        details += f"**片段 {i + 1}（相似度：{score}）：**\n{preview}\n\n"
    return details

# -------------------------- 全局初始化 --------------------------
chunks = load_and_split_document()
retriever = TfidfRetriever(chunks)

logger.info("正在初始化豆包大模型...")
try:
    client = Ark(api_key=ARK_API_KEY, base_url=BASE_URL)
except Exception as e:
    logger.error("初始化豆包失败：%s", e)
    print(f"❌ 初始化豆包失败：{e}")
    print("   请检查 config.py 中的 API Key 是否正确")
    raise SystemExit(1) from e
logger.info("豆包大模型初始化完成")

PROMPT_TEMPLATE = """
你是巢湖流域污染分析项目的专业问答助手。
**绝对规则：**
1. 只能使用下面提供的上下文内容回答问题，绝对不能使用你自己的任何知识
2. 如果上下文里没有答案，直接说："抱歉，论文中没有提到相关内容"
3. 回答要准确、简洁、专业，符合学术规范，分点列出关键信息
4. 不要添加任何推测和解释，只陈述论文中的事实

上下文内容：
{context}

用户问题：{question}

回答：
"""

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>巢湖流域污染分析智能问答系统</title>
    <style>
        body { max-width: 1000px; margin: 0 auto; padding: 20px; font-family: Arial, sans-serif; }
        h1 { color: #2c3e50; text-align: center; }
        .info { background: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
        .chat-box { border: 1px solid #ddd; border-radius: 5px; padding: 20px; height: 400px; overflow-y: auto; margin-bottom: 20px; }
        .message { margin-bottom: 15px; padding: 10px; border-radius: 5px; }
        .user { background: #f0f0f0; white-space: pre-wrap; }
        .assistant { background: #e3f2fd; line-height: 1.6; }
        .assistant h3 { margin: 12px 0 8px; color: #2c3e50; font-size: 1em; }
        .assistant hr { border: none; border-top: 1px solid #ccc; margin: 12px 0; }
        .chunk-preview { font-size: 0.9em; color: #555; margin-bottom: 10px; }
        .input-area { display: flex; gap: 10px; }
        input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:disabled { background: #bdc3c7; cursor: not-allowed; }
        .preset { margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap; }
        .preset button { background: #95a5a6; }
    </style>
</head>
<body>
    <h1>🌊 巢湖流域污染分析智能问答系统</h1>
    <div class="info">
        <strong>技术说明：</strong>基于RAG技术 | TF-IDF向量检索 | 内存缓存 | 检索可解释 | 杜绝幻觉
    </div>
    <div class="chat-box" id="chatBox"></div>
    <div class="input-area">
        <input type="text" id="question" placeholder="请输入你的问题...">
        <button id="sendBtn" onclick="askQuestion()">发送</button>
    </div>
    <div class="preset">
        <button onclick="askPreset('本次研究的背景和意义是什么？')">研究背景</button>
        <button onclick="askPreset('巢湖流域17个区县的污染聚类结果是什么？')">聚类结果</button>
        <button onclick="askPreset('高污染区域主要分布在哪些地方？')">污染分布</button>
        <button onclick="askPreset('研究使用了哪些分析方法？')">研究方法</button>
        <button onclick="askPreset('本次研究的核心结论是什么？')">核心结论</button>
    </div>

    <script>
        function renderMarkdown(text) {
            return text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/^### (.*)$/gm, '<h3>$1</h3>')
                .replace(/^---$/gm, '<hr>')
                .replace(/\\n/g, '<br>');
        }

        function addMessage(text, isUser) {
            const chatBox = document.getElementById('chatBox');
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user' : 'assistant');

            if (isUser) {
                div.textContent = text;
            } else {
                div.innerHTML = renderMarkdown(text);
            }

            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function askQuestion() {
            const question = document.getElementById('question').value.trim();
            if (!question) return;

            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = true;

            addMessage(question, true);
            document.getElementById('question').value = '';
            addMessage('正在检索论文并生成回答，请稍候...', false);

            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: 'question=' + encodeURIComponent(question)
                });
                const answer = await res.text();
                const chatBox = document.getElementById('chatBox');
                chatBox.removeChild(chatBox.lastChild);
                addMessage(answer, false);
            } catch (err) {
                const chatBox = document.getElementById('chatBox');
                chatBox.removeChild(chatBox.lastChild);
                addMessage('网络请求失败，请检查服务是否正常运行。', false);
            } finally {
                sendBtn.disabled = false;
            }
        }

        function askPreset(question) {
            document.getElementById('question').value = question;
            askQuestion();
        }

        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') askQuestion();
        });
    </script>
</body>
</html>
"""


class RAGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path != "/ask":
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")
            params = parse_qs(post_data)
            question = params.get("question", [""])[0].strip()

            if not question:
                self._send_text(400, "请输入有效的问题。")
                return

            logger.info("用户提问：%s", question)

            cached_answer = _cache_get(question)
            if cached_answer is not None:
                logger.info("从缓存返回回答：%s", question)
                full_answer = cached_answer
            else:
                retrieval_results = retriever.retrieve(question, TOP_K)
                context = "\n\n".join(chunk for chunk, _ in retrieval_results)
                final_prompt = PROMPT_TEMPLATE.format(context=context, question=question)

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

                answer = response.choices[0].message.content
                full_answer = (
                    f"{answer}{_format_retrieval_details(retrieval_results)}"
                    f"---\n来源：论文原文"
                )
                _cache_set(question, full_answer)
                logger.info("生成新回答：%s", question)

            self._send_text(200, full_answer)

        except Exception as e:
            logger.error("处理请求失败：%s", str(e), exc_info=True)
            self._send_text(500, "抱歉，服务器暂时无法回答你的问题，请稍后再试。")

    def _send_text(self, status, text):
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))


if __name__ == "__main__":
    print("🚀 系统正在启动...")
    print(f"🌐 请在浏览器中打开：http://localhost:{PORT}")
    print("✅ 系统启动成功！按 Ctrl+C 停止服务")
    logger.info("系统启动成功，监听 %s:%d", HOST, PORT)

    server = HTTPServer((HOST, PORT), RAGHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 系统已停止")
        logger.info("系统已停止")
        server.shutdown()
