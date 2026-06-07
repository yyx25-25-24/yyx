import math
import re
from collections import Counter

# 中文常见停用词：降低「的、是、在」等词的权重
CHINESE_STOP_WORDS = {
    "的", "是", "在", "了", "和", "与", "及", "等", "为", "有", "对", "中",
    "上", "下", "这", "那", "其", "之", "以", "将", "从", "被", "所", "而",
    "或", "也", "都", "要", "一个", "我们", "他们", "本文", "进行", "通过",
    "采用", "具有", "作为", "以及", "其中", "因此", "由于", "但是", "如果",
    "不是", "没有", "可能", "已经", "主要", "相关", "不同", "这些", "那些",
}


def _tokenize(text):
    """中英文混合分词：单字/单词 + 相邻中文二元组（如「巢湖」）"""
    units = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text.lower())
    tokens = []

    for i, unit in enumerate(units):
        if unit not in CHINESE_STOP_WORDS:
            tokens.append(unit)

        if (
            len(unit) == 1
            and "\u4e00" <= unit <= "\u9fff"
            and i + 1 < len(units)
            and len(units[i + 1]) == 1
            and "\u4e00" <= units[i + 1] <= "\u9fff"
        ):
            bigram = unit + units[i + 1]
            if bigram not in CHINESE_STOP_WORDS:
                tokens.append(bigram)

    return tokens


def _tfidf_vector(tokens, idf):
    """计算单个文本的 TF-IDF 向量（sublinear TF）"""
    if not tokens:
        return {}

    tf = Counter(tokens)
    vector = {}
    for term, count in tf.items():
        if term in idf:
            vector[term] = (1 + math.log(count)) * idf[term]
    return vector


def _cosine_similarity(vec_a, vec_b):
    """余弦相似度"""
    if not vec_a or not vec_b:
        return 0.0

    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TfidfRetriever:
    """
    TF-IDF 向量检索引擎（纯 Python 实现，无需 sklearn/scipy）
    原理：将文本转换为向量，通过余弦相似度找到最相关的文本块
    """

    def __init__(self, chunks):
        self.chunks = chunks
        print("🔢 正在构建 TF-IDF 向量矩阵...")

        self.doc_tokens = [_tokenize(chunk) for chunk in chunks]
        n_docs = len(chunks)

        doc_freq = Counter()
        for tokens in self.doc_tokens:
            doc_freq.update(set(tokens))

        self.idf = {
            term: math.log((n_docs + 1) / (df + 1)) + 1
            for term, df in doc_freq.items()
        }
        self.vectors = [_tfidf_vector(tokens, self.idf) for tokens in self.doc_tokens]

        print("✅ TF-IDF 向量矩阵构建完成")

    def retrieve(self, query, top_k=3):
        """检索最相关的 top_k 个文本块，返回 [(文本块, 相似度), ...]"""
        if not self.chunks:
            return []

        query_vec = _tfidf_vector(_tokenize(query), self.idf)
        scores = [
            (_cosine_similarity(query_vec, vec), idx)
            for idx, vec in enumerate(self.vectors)
        ]
        scores.sort(reverse=True, key=lambda x: x[0])
        return [
            (self.chunks[idx], round(score, 4))
            for score, idx in scores[:top_k]
        ]
