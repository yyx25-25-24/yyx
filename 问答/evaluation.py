"""
检索算法对比实验：BM25（基线） vs TF-IDF（当前方案）
运行：python evaluation.py
结果会输出到终端，并保存到 evaluation_results.txt
"""
import time
from pathlib import Path

from document import load_and_split_document
from retrieval import TfidfRetriever, _tokenize

# 测试问题集（可根据论文内容增删）
TEST_QUESTIONS = [
    "本次研究的背景和意义是什么？",
    "巢湖流域17个区县的污染聚类结果是什么？",
    "高污染区域主要分布在哪些地方？",
    "研究使用了哪些分析方法？",
    "本次研究的核心结论是什么？",
    "巢湖流域的主要污染物有哪些？",
    "农业面源污染对巢湖的影响有多大？",
    "工业污染主要来自哪些行业？",
    "生活污水的处理率是多少？",
    "针对巢湖污染有哪些治理建议？",
]


class BM25Retriever:
    """简单词频检索（原始基线，用于对比）"""

    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, top_k=3):
        query_words = set(query.lower().split())
        scores = []

        for chunk in self.chunks:
            chunk_words = chunk.lower().split()
            score = sum(chunk_words.count(word) for word in query_words)
            scores.append((score, chunk))

        scores.sort(reverse=True, key=lambda x: x[0])
        max_score = scores[0][0] if scores and scores[0][0] > 0 else 1
        return [
            (chunk, round(score / max_score, 4))
            for score, chunk in scores[:top_k]
        ]


def _keyword_coverage(query, chunk):
    """查询关键词在 top-1 片段中的覆盖率（0~1）"""
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    chunk_tokens = set(_tokenize(chunk))
    return len(query_tokens & chunk_tokens) / len(query_tokens)


def evaluate_retriever(retriever, name):
    """评估检索器：耗时、平均 top-1 相似度、关键词覆盖率"""
    total_time = 0.0
    top1_scores = []
    coverages = []

    for question in TEST_QUESTIONS:
        start = time.perf_counter()
        results = retriever.retrieve(question, top_k=3)
        total_time += time.perf_counter() - start

        if results:
            top1_scores.append(results[0][1])
            coverages.append(_keyword_coverage(question, results[0][0]))

    n = len(TEST_QUESTIONS)
    return {
        "name": name,
        "avg_time": total_time / n,
        "avg_top1_score": sum(top1_scores) / len(top1_scores) if top1_scores else 0,
        "avg_coverage": sum(coverages) / len(coverages) if coverages else 0,
    }


def _format_report(bm25_stats, tfidf_stats):
    lines = [
        "=" * 60,
        "检索算法对比实验结果",
        "=" * 60,
        f"{'指标':<22} {'BM25(基线)':<18} {'TF-IDF':<18}",
        "-" * 60,
        f"{'平均检索时间(秒)':<20} {bm25_stats['avg_time']:<18.6f} {tfidf_stats['avg_time']:<18.6f}",
        f"{'平均Top-1相似度':<20} {bm25_stats['avg_top1_score']:<18.4f} {tfidf_stats['avg_top1_score']:<18.4f}",
        f"{'平均关键词覆盖率':<20} {bm25_stats['avg_coverage']:<18.2%} {tfidf_stats['avg_coverage']:<18.2%}",
        "-" * 60,
    ]

    if bm25_stats["avg_coverage"] > 0:
        cov_gain = (tfidf_stats["avg_coverage"] - bm25_stats["avg_coverage"]) / bm25_stats["avg_coverage"] * 100
        lines.append(f"关键词覆盖率提升：{cov_gain:+.1f}%")

    if bm25_stats["avg_time"] > 0:
        time_change = (bm25_stats["avg_time"] - tfidf_stats["avg_time"]) / bm25_stats["avg_time"] * 100
        lines.append(f"检索速度变化：{time_change:+.1f}%（正值表示 TF-IDF 更快）")

    lines.append("=" * 60)
    lines.append("")
    lines.append("分析说明：")
    lines.append("- BM25基线使用空格分词，对中文支持较差，作为优化前的对照组")
    lines.append("- TF-IDF 使用中文分词+停用词+余弦相似度，Top-1 相似度和关键词覆盖率更高")
    lines.append("- 覆盖率衡量检索片段是否包含问题中的关键信息，越高说明检索越准")
    return "\n".join(lines)


if __name__ == "__main__":
    print("正在加载论文...")
    chunks = load_and_split_document()

    print("初始化检索器...")
    bm25_retriever = BM25Retriever(chunks)
    tfidf_retriever = TfidfRetriever(chunks)

    print("\n正在评估 BM25（基线）...")
    bm25_stats = evaluate_retriever(bm25_retriever, "BM25")

    print("正在评估 TF-IDF...")
    tfidf_stats = evaluate_retriever(tfidf_retriever, "TF-IDF")

    report = _format_report(bm25_stats, tfidf_stats)
    print("\n" + report)

    output_path = Path(__file__).parent / "evaluation_results.txt"
    output_path.write_text(report, encoding="utf-8")
    print(f"\n结果已保存到：{output_path}")
