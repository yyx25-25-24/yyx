"""
通用文档问答系统测评模块
运行：python evaluation_system.py
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from app import (
    parse_text_from_file,
    generate_answer,
    retrieve_chunks_tfidf,
    _tokenize,
    llm_client,
    ARK_API_KEY,
)

TEST_QUESTIONS = [
    {"question": "文档的主要内容是什么？", "type": "simple"},
    {"question": "有哪些关键概念？", "type": "simple"},
    {"question": "结论是什么？", "type": "simple"},
    {"question": "请总结文档的核心观点，并分析其逻辑结构", "type": "complex"},
    {"question": "文档中提到的主要问题和解决方案分别是什么？它们之间有什么关联？", "type": "complex"},
    {"question": "对比文档中的不同观点或方法，分析它们的优缺点", "type": "complex"},
    {"question": "文档作者的个人背景是什么？", "type": "tricky"},
    {"question": "这篇文档发表于哪一年？", "type": "tricky"},
    {"question": "文档中提到的具体数据指标是多少？", "type": "tricky"},
]


def _check_hallucination(answer, context):
    answer_tokens = set(_tokenize(answer))
    context_tokens = set(_tokenize(context))
    
    important_words = {t for t in answer_tokens if len(t) >= 2 and t not in {"的", "是", "在", "了"}}
    
    if not important_words:
        return 0.0
    
    unknown_words = important_words - context_tokens
    hallucination_rate = len(unknown_words) / len(important_words)
    
    return min(hallucination_rate, 1.0)


def _estimate_token_cost(prompt_length, answer_length):
    total_chars = prompt_length + answer_length
    estimated_tokens = int(total_chars * 1.5)
    cost = estimated_tokens / 1000 * 0.01
    return cost, estimated_tokens


def evaluate_on_document(document_path, document_name):
    print(f"\n{'='*60}")
    print(f"测评文档：{document_name}")
    print(f"{'='*60}")
    
    document_text = parse_text_from_file(document_path)
    if not document_text:
        print("❌ 无法解析文档")
        return None
    
    results = []
    total_time = 0.0
    total_cost = 0.0
    total_tokens = 0
    hallucination_scores = []
    
    for item in TEST_QUESTIONS:
        question = item["question"]
        q_type = item["type"]
        
        print(f"\n测试问题 [{q_type}]：{question}")
        
        start = time.perf_counter()
        
        answer = generate_answer(document_text, question, file_key=document_name)
        
        elapsed = time.perf_counter() - start
        total_time += elapsed
        
        retrieval_results = retrieve_chunks_tfidf(document_text, question)
        context = "\n\n".join(chunk for chunk, _ in retrieval_results) if retrieval_results else ""
        
        cost, tokens = _estimate_token_cost(len(question), len(answer))
        total_cost += cost
        total_tokens += tokens
        
        hall_score = _check_hallucination(answer, context)
        hallucination_scores.append(hall_score)
        
        print(f"  ⏱️  耗时：{elapsed:.3f}s")
        print(f"  💰 成本：¥{cost:.4f}")
        print(f"  👻 幻觉率：{hall_score:.2%}")
        
        results.append({
            "question": question,
            "type": q_type,
            "time": elapsed,
            "cost": cost,
            "tokens": tokens,
            "hallucination": hall_score,
            "answer_preview": answer[:200],
        })
    
    n = len(TEST_QUESTIONS)
    stats = {
        "document": document_name,
        "avg_time": total_time / n,
        "avg_cost": total_cost / n,
        "total_tokens": total_tokens,
        "avg_hallucination": sum(hallucination_scores) / n if hallucination_scores else 0,
        "details": results,
    }
    
    return stats


def _format_report(all_stats):
    lines = [
        "=" * 80,
        "通用文档问答系统测评报告",
        "=" * 80,
        "",
    ]
    
    for stats in all_stats:
        lines.extend([
            f"文档：{stats['document']}",
            "-" * 80,
            f"平均响应时延：{stats['avg_time']:.3f}秒",
            f"平均单轮成本：¥{stats['avg_cost']:.4f}",
            f"总Token消耗：{stats['total_tokens']}",
            f"平均幻觉率：{stats['avg_hallucination']:.2%}",
            "",
            "各问题类型表现：",
        ])
        
        type_stats = {}
        for detail in stats["details"]:
            q_type = detail["type"]
            if q_type not in type_stats:
                type_stats[q_type] = {"count": 0, "hallucination": [], "time": []}
            type_stats[q_type]["count"] += 1
            type_stats[q_type]["hallucination"].append(detail["hallucination"])
            type_stats[q_type]["time"].append(detail["time"])
        
        type_names = {"simple": "简单题", "complex": "复杂题", "tricky": "易错题"}
        for q_type, t_stats in type_stats.items():
            avg_hall = sum(t_stats["hallucination"]) / len(t_stats["hallucination"])
            avg_time = sum(t_stats["time"]) / len(t_stats["time"])
            lines.append(f"  {type_names.get(q_type, q_type):<10} 幻觉率:{avg_hall:.2%}  时延:{avg_time:.3f}s")
        
        lines.append("")
    
    lines.extend([
        "=" * 80,
        "说明：",
        "- 幻觉率衡量回答中出现文档中不存在信息的比例",
        "- 成本基于API定价估算（每1000 token约¥0.01）",
        "- 建议对多个不同类型的文档进行测试以获得全面评估",
        "=" * 80,
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    if not ARK_API_KEY:
        print("⚠️ 未配置API密钥，将仅测试本地检索模式")
    
    upload_folder = Path(__file__).parent / "uploads"
    documents = list(upload_folder.glob("*"))
    
    if not documents:
        print("❌ uploads文件夹中没有文档，请先上传文件")
        print("   启动系统后访问 http://localhost:5000/upload 上传文档")
        sys.exit(1)
    
    print(f"📊 发现 {len(documents)} 个文档，开始测评...")
    
    all_stats = []
    for doc_path in documents:
        stats = evaluate_on_document(str(doc_path), doc_path.name)
        if stats:
            all_stats.append(stats)
    
    if all_stats:
        report = _format_report(all_stats)
        print("\n" + report)
        
        output_path = Path(__file__).parent / "evaluation_results.txt"
        output_path.write_text(report, encoding="utf-8")
        print(f"\n✅ 结果已保存到：{output_path}")
    else:
        print("❌ 没有生成任何测评结果")