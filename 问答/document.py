import sys

import docx

from config import CHUNK_SIZE, DOCUMENT_PATH


def load_and_split_document():
    """加载Word文档并进行智能语义分块"""
    print("📄 正在加载论文...")

    try:
        doc = docx.Document(DOCUMENT_PATH)
    except FileNotFoundError:
        print(f"❌ 错误：找不到论文文件 {DOCUMENT_PATH}")
        print("   请检查 config.py 中的 DOCUMENT_PATH 是否正确")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载论文失败：{e}")
        sys.exit(1)

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        print("❌ 错误：论文文件中没有可读取的文本内容")
        sys.exit(1)

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        current_chunk.append(para)
        current_length += len(para)

        if current_length > CHUNK_SIZE:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    print(f"✅ 论文处理完成，共生成 {len(chunks)} 个语义块")
    return chunks
