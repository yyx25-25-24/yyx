import os
import time
from pathlib import Path
from collections import OrderedDict
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
from collections import OrderedDict, Counter

# 导入我们刚刚拆分的检索模块
from retrieval import retrieve_chunks_tfidf, tokenize, cosine_similarity

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"txt", "pdf", "md", "docx"}
TOP_K = 3
PAGE_SIZE = 10
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

ARK_API_KEY = os.getenv("ARK_API_KEY", "")
MODEL_NAME = "doubao-seed-1-8-251228"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
TEMPERATURE = 0.01
MAX_TOKENS = 1024
MAX_CACHE_SIZE = 100
SIMILARITY_THRESHOLD = 0.9
SIMPLE_QUESTION_MAX_LENGTH = 20


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_text_from_file(path: str) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            texts = []
            for p in reader.pages:
                text = p.extract_text() or ""
                texts.append(text)
            return "\n".join(texts)
        except Exception as e:
            print(f"PDF解析错误 {path.name}: {e}")
            return ""
    elif suffix == ".docx":
        try:
            doc = Document(str(path))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        except Exception as e:
            print(f"Docx解析错误 {path.name}: {e}")
            return ""
    elif suffix in {".txt", ".md"}:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"Txt解析错误 {path.name}: {e}")
            return ""
    return ""


def _is_simple_question(question: str) -> bool:
    if len(question) >= SIMPLE_QUESTION_MAX_LENGTH:
        return False
    
    simple_keywords = [
        "是什么", "有哪些", "在哪里", "什么时候", "谁", 
        "多少", "几个", "叫什么", "目录", "标题", "章节"
    ]
    
    has_simple_keyword = any(keyword in question for keyword in simple_keywords)
    structure_keywords = ["目录", "章节", "标题", "结构", "大纲"]
    is_structure_query = any(keyword in question for keyword in structure_keywords)
    
    return len(question) < SIMPLE_QUESTION_MAX_LENGTH and (has_simple_keyword or is_structure_query)


class SimilarQuestionCache:
    def __init__(self, max_size=100, similarity_threshold=0.9):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache = []
    
    def _vectorize_question(self, question):
        tokens = tokenize(question)
        return dict(Counter(tokens))
    
    def find_similar(self, question):
        if not self.cache:
            return None
        
        question_vec = self._vectorize_question(question)
        
        for cached_vec, cached_question, cached_answer in self.cache:
            similarity = cosine_similarity(question_vec, cached_vec)
            if similarity >= self.similarity_threshold:
                return cached_answer
        
        return None
    
    def add(self, question, answer):
        question_vec = self._vectorize_question(question)
        self.cache.append((question_vec, question, answer))
        
        if len(self.cache) > self.max_size:
            self.cache.pop(0)


answer_cache = OrderedDict()
similar_question_cache = SimilarQuestionCache(max_size=MAX_CACHE_SIZE, similarity_threshold=SIMILARITY_THRESHOLD)

llm_client = None
if ARK_API_KEY:
    try:
        from volcenginesdkarkruntime import Ark
        llm_client = Ark(api_key=ARK_API_KEY, base_url=BASE_URL)
        print("✅ LLM客户端初始化成功")
    except Exception as e:
        print(f"⚠️ LLM初始化失败：{e}，将使用本地检索模式")


PROMPT_TEMPLATE = """
你是专业的文档分析助手。你将收到来自一个或多个文档的上下文内容。

**回答规则：**
1. **基于事实**：你的回答必须严格基于提供的上下文内容。
2. **综合分析与对比**：如果用户要求对比，请仔细分析不同来源的信息，找出异同点。
3. **多文档处理**：上下文标记为 "=== 文档来源: xxx ==="。请注意区分信息来源。
4. **诚实回答**：如果上下文中没有相关信息，请说明。

**上下文内容：**
{context}

**用户问题：**
{question}

**请开始回答：**
"""


def _call_llm_direct(context: str, question: str) -> str:
    if not llm_client:
        return "⚠️ 未配置 LLM API。"
    try:
        final_prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM 调用失败：{e}"


def generate_answer(document_text: str, question: str, file_key: str = None) -> str:
    if not document_text:
        return "(空文档或无法解析)"
    if not question.strip():
        return document_text[:1000]
    
    cache_key = f"{file_key}:{question}" if file_key else question
    if cache_key in answer_cache:
        answer_cache.move_to_end(cache_key)
        return answer_cache[cache_key]
    
    similar_answer = similar_question_cache.find_similar(question)
    if similar_answer is not None:
        answer_cache[cache_key] = similar_answer
        if len(answer_cache) > MAX_CACHE_SIZE:
            answer_cache.popitem(last=False)
        return similar_answer
    
    is_simple = _is_simple_question(question)
    if is_simple:
        retrieval_results = retrieve_chunks_tfidf(document_text, question, TOP_K)
        if not retrieval_results:
            return document_text[:1000]
        context = "\n\n".join(chunk for chunk, _ in retrieval_results)
        answer = f"根据文档内容：\n\n{context}"
    else:
        retrieval_results = retrieve_chunks_tfidf(document_text, question, TOP_K)
        if not retrieval_results:
            return document_text[:1000]
        context = "\n\n".join(chunk for chunk, _ in retrieval_results)
        if llm_client:
            try:
                final_prompt = PROMPT_TEMPLATE.format(context=context, question=question)
                response = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"⚠️ LLM调用失败：{e}"
        else:
            answer = f"根据文档内容：\n\n{context}"
    
    answer_cache[cache_key] = answer
    if len(answer_cache) > MAX_CACHE_SIZE:
        answer_cache.popitem(last=False)
    similar_question_cache.add(question, answer)
    return answer


def simple_qa(document_text: str, question: str) -> str:
    return generate_answer(document_text, question)


def list_uploaded_files():
    uploads = sorted(os.listdir(UPLOAD_FOLDER))
    files = []
    for f in uploads:
        p = os.path.join(UPLOAD_FOLDER, f)
        try:
            stat = os.stat(p)
            files.append({"name": f, "path": p, "size": stat.st_size, "mtime": stat.st_mtime})
        except Exception:
            continue
    return files


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-in-production")


@app.route("/")
def index():
    return redirect(url_for("upload_file"))


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("未找到上传文件字段")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("未选择文件")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            dest = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(dest)
            flash("上传成功")
            return redirect(url_for("file_list"))
        else:
            flash("不支持的文件类型")
            return redirect(request.url)
    uploads = sorted(os.listdir(app.config["UPLOAD_FOLDER"]))
    return render_template("upload.html", uploads=uploads)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/files")
def file_list():
    q = request.args.get("q", "").strip()
    page = int(request.args.get("page", "1") or "1")
    files = list_uploaded_files()
    if q:
        q_low = q.lower()
        filtered = []
        for info in files:
            if q_low in info["name"].lower():
                filtered.append(info)
                continue
            try:
                text = parse_text_from_file(info["path"]) or ""
                if q_low in text.lower():
                    filtered.append(info)
            except Exception:
                continue
        files = filtered
    total = len(files)
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_files = files[start:end]
    for f in page_files:
        f["size_kb"] = f"{f['size']//1024} KB"
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE if total > 0 else 1
    return render_template("files.html", uploads=page_files, q=q, page=page, total=total, total_pages=total_pages)


@app.route("/qa", methods=["GET", "POST"])
def qa():
    uploads = list_uploaded_files()
    answer = None
    selected_file = ""
    question = ""
    
    if request.method == "POST":
        selected_file = request.form.get("files") 
        question = request.form.get("question", "")
        
        if not selected_file:
            flash("请选择一个文档")
        else:
            path = os.path.join(app.config["UPLOAD_FOLDER"], selected_file)
            if not os.path.exists(path):
                answer = "❌ 文件不存在。"
            else:
                text = parse_text_from_file(path)
                print(f"📄 文件 [{selected_file}] 解析长度: {len(text)} 字符")
                
                if not text or len(text.strip()) < 10:
                    answer = f"⚠️ 文件 {selected_file} 内容为空或无法解析（可能是扫描版PDF）。"
                else:
                    answer = generate_answer(text, question, file_key=selected_file)
    
    upload_names = [f['name'] for f in uploads]
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'answer': answer,
            'status': 'success'
        })
    
    return render_template("qa.html", uploads=upload_names, answer=answer, selected_files=[selected_file], question=question)


@app.route('/delete', methods=['POST'])
def delete_files():
    files = request.form.getlist('files')
    if not files:
        f = request.form.get('files')
        if f: files = [f]
    deleted = []
    for name in files:
        safe = os.path.basename(name)
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
        if os.path.exists(path):
            try:
                os.remove(path)
                deleted.append(safe)
            except: pass
    if deleted: flash(f"已删除: {', '.join(deleted)}")
    return redirect(url_for('file_list'))


@app.route('/preview')
def preview():
    filename = request.args.get('file')
    if not filename: return redirect(url_for('file_list'))
    safe = os.path.basename(filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
    if not os.path.exists(path): return redirect(url_for('file_list'))
    content = parse_text_from_file(path) or '(无法解析)'
    return render_template('preview.html', filename=safe, content=content)


if __name__ == "__main__":
    print("🚀 系统启动...")
    app.run(host="0.0.0.0", port=5000, debug=True)