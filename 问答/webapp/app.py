import os
import re
from pathlib import Path
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"txt", "pdf", "md", "docx"}

TOP_K = 3
PAGE_SIZE = 10


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
        except Exception:
            return ""
    elif suffix == ".docx":
        try:
            doc = Document(str(path))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        except Exception:
            return ""
    elif suffix in {".txt", ".md"}:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    return ""


def split_into_chunks(text: str, chunk_size: int = 800) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    current = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) + 1 > chunk_size and current:
            chunks.append("\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def retrieve_chunks(document_text: str, question: str, top_k: int = TOP_K) -> list[tuple[str, int]]:
    chunks = split_into_chunks(document_text)
    keywords = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2]
    scored = []
    for chunk in chunks:
        score = sum(chunk.lower().count(k) for k in keywords)
        if score > 0:
            scored.append((chunk, score))
    if not scored:
        # fallback return the first few chunks if no keyword match
        return [(chunk, 0) for chunk in chunks[:top_k]]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def list_uploaded_files():
    uploads = sorted(os.listdir(UPLOAD_FOLDER))
    files = []
    for f in uploads:
        p = os.path.join(UPLOAD_FOLDER, f)
        try:
            stat = os.stat(p)
            files.append({
                "name": f,
                "path": p,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
        except Exception:
            continue
    return files


PROMPT_TEMPLATE = """
你是文档问答助手。
严格规则：
1. 只能使用下面提供的上下文回答问题。
2. 如果上下文中没有答案，直接回答“抱歉，文档中没有提到相关内容”。
3. 回答要准确、简洁、专业。

上下文内容：
{context}

用户问题：{question}

回答：
"""


def simple_qa(document_text: str, question: str) -> str:
    if not document_text:
        return "(空文档或无法解析)"
    if not question.strip():
        return document_text[:1000]
    retrieval_results = retrieve_chunks(document_text, question)
    if not retrieval_results:
        return document_text[:1000]
    return "\n---\n".join(chunk for chunk, _ in retrieval_results)

# 本地模式：始终使用简单检索问答，不调用任何外部大模型


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "dev-secret"


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
    # 支持 filename/content 搜索与分页
    q = request.args.get("q", "").strip()
    page = int(request.args.get("page", "1") or "1")

    files = list_uploaded_files()

    # 如果有查询，按文件名或内容过滤
    if q:
        q_low = q.lower()
        filtered = []
        for info in files:
            if q_low in info["name"].lower():
                filtered.append(info)
                continue
            # 内容搜索（解析文本，可能较慢）
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

    # 用于模板显示简短信息
    for f in page_files:
        f["size_kb"] = f"{f['size']//1024} KB"

    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE if total > 0 else 1

    return render_template("files.html", uploads=page_files, q=q, page=page, total=total, total_pages=total_pages)


@app.route("/qa", methods=["GET", "POST"])
def qa():
    uploads = sorted(os.listdir(app.config["UPLOAD_FOLDER"]))
    answer = None
    selected = None
    if request.method == "POST":
        selected = request.form.get("file")
        question = request.form.get("question", "")
        if not selected or selected not in [f['name'] for f in uploads]:
            flash("请选择有效的文件")
        else:
            path = os.path.join(app.config["UPLOAD_FOLDER"], selected)
            text = parse_text_from_file(path)
            answer = simple_qa(text, question)
    return render_template("qa.html", uploads=uploads, answer=answer, selected=selected)


@app.route('/delete', methods=['POST'])
def delete_files():
    # 支持单个或多个 files 参数
    files = request.form.getlist('files')
    if not files:
        # 也尝试单个字段
        f = request.form.get('files')
        if f:
            files = [f]

    deleted = []
    failed = []
    for name in files:
        safe = os.path.basename(name)
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
        if os.path.exists(path):
            try:
                os.remove(path)
                deleted.append(safe)
            except Exception:
                failed.append(safe)
        else:
            failed.append(safe)

    if deleted:
        flash(f"已删除: {', '.join(deleted)}")
    if failed and not deleted:
        flash(f"未删除任何文件，失败: {', '.join(failed)}")
    elif failed:
        flash(f"部分失败，未能删除: {', '.join(failed)}")

    return redirect(url_for('file_list'))


@app.route('/preview')
def preview():
    filename = request.args.get('file')
    if not filename:
        flash('缺少文件名')
        return redirect(url_for('file_list'))
    safe = os.path.basename(filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
    if not os.path.exists(path):
        flash('文件不存在')
        return redirect(url_for('file_list'))

    content = parse_text_from_file(path)
    if not content:
        content = '(无法解析或空文件)'

    return render_template('preview.html', filename=safe, content=content)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
