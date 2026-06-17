import re
import math
from collections import Counter

def tokenize(text):
    """
    对文本进行分词，支持中文单字+二元组和英文单词。
    """
    units = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text.lower())
    tokens = []
    stop_words = {"的", "是", "在", "了", "和", "与", "及", "等", "为", "有", "对", "中"}
    
    for i, unit in enumerate(units):
        if unit not in stop_words:
            tokens.append(unit)
        
        if (
            len(unit) == 1
            and "\u4e00" <= unit <= "\u9fff"
            and i + 1 < len(units)
            and len(units[i + 1]) == 1
            and "\u4e00" <= units[i + 1] <= "\u9fff"
        ):
            bigram = unit + units[i + 1]
            if bigram not in stop_words:
                tokens.append(bigram)
    
    return tokens

def cosine_similarity(vec_a, vec_b):
    """
    计算两个向量之间的余弦相似度。
    """
    if not vec_a or not vec_b:
        return 0.0
    
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def split_into_chunks(text: str, chunk_size: int = 800) -> list[str]:
    """
    将长文本按段落分割成固定大小的片段。
    """
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

def retrieve_chunks_tfidf(document_text: str, question: str, top_k: int = 3) -> list[tuple[str, float]]:
    """
    基于 TF-IDF 算法检索与问题最相关的文本片段。
    """
    chunks = split_into_chunks(document_text)
    if not chunks:
        return []
    
    doc_tokens = [tokenize(chunk) for chunk in chunks]
    n_docs = len(chunks)
    doc_freq = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))
    
    idf = {
        term: math.log((n_docs + 1) / (df + 1)) + 1
        for term, df in doc_freq.items()
    }
    
    query_tokens = tokenize(question)
    query_tf = Counter(query_tokens)
    query_vec = {
        term: (1 + math.log(count)) * idf.get(term, 0)
        for term, count in query_tf.items()
    }
    
    scores = []
    for chunk, tokens in zip(chunks, doc_tokens):
        tf = Counter(tokens)
        chunk_vec = {
            term: (1 + math.log(count)) * idf.get(term, 0)
            for term, count in tf.items()
        }
        similarity = cosine_similarity(query_vec, chunk_vec)
        if similarity > 0:
            scores.append((chunk, round(similarity, 4)))
    
    if not scores:
        return [(chunk, 0) for chunk in chunks[:top_k]]
    
    scores.sort(reverse=True, key=lambda x: x[1])
    return scores[:top_k]