import os
import datetime
from pathlib import Path


def save_markdown_report(query: str, response: str, dataset_key: str, save_dir: str, charts: list[str] | None = None) -> str:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{dataset_key}_{timestamp}.md"
    filename = filename.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"# 分析报告\n\n")
        f.write(f"- 数据集：{dataset_key}\n")
        f.write(f"- 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 查询内容：{query}\n\n")
        f.write("---\n\n")
        f.write("## 分析结果\n\n")
        # 如果 response 是可解析的 JSON 字符串或 dict，尝试格式化为表格
        parsed = None
        if isinstance(response, dict):
            parsed = response
        else:
            try:
                parsed = json.loads(response)
            except Exception:
                parsed = None

        if isinstance(parsed, dict):
            # 若包含多个表格型字段，逐个导出为 markdown 表格
            for k, v in parsed.items():
                if isinstance(v, list):
                    # list of records -> table
                    try:
                        from pandas import DataFrame
                        df = DataFrame(v)
                        f.write(f"### {k}\n")
                        f.write(df.to_markdown(index=False))
                        f.write("\n\n")
                        continue
                    except Exception:
                        pass
                # 其他字段直接写入
                f.write(f"**{k}**:\n{v}\n\n")
        else:
            f.write(response)

        # 嵌入图表（如果提供）
        if charts:
            f.write("## 图表\n\n")
            for c in charts:
                # 如果是 html，则以链接形式嵌入；如果是图片则显示图片
                ext = os.path.splitext(c)[1].lower()
                if ext in (".html",):
                    f.write(f"- 图表文件：[{os.path.basename(c)}]({c})\n")
                else:
                    # 使用 markdown 图片展示
                    f.write(f"![{os.path.basename(c)}]({c})\n\n")
        f.write("\n")

    return save_path


def save_docx_report(query: str, response: str, dataset_key: str, save_dir: str) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("缺少 python-docx 库，请先安装：pip install python-docx") from exc

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{dataset_key}_{timestamp}.docx"
    filename = filename.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(save_dir, filename)

    doc = Document()
    doc.add_heading("分析报告", level=1)
    doc.add_paragraph(f"数据集：{dataset_key}")
    doc.add_paragraph(f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"查询内容：{query}")
    doc.add_paragraph("")
    doc.add_heading("分析结果", level=2)
    for line in response.splitlines():
        doc.add_paragraph(line)

    doc.save(save_path)
    return save_path


def save_analysis_report(query: str, response: str, dataset_key: str, save_dir: str, report_format: str) -> str:
    if report_format.lower() == "md":
        # backward-compatible: accept optional charts passed via tuple
        charts = None
        if isinstance(response, tuple) and len(response) == 2:
            resp_text, charts = response
            return save_markdown_report(query, resp_text, dataset_key, save_dir, charts=charts)
        return save_markdown_report(query, response, dataset_key, save_dir)
    if report_format.lower() == "docx":
        return save_docx_report(query, response, dataset_key, save_dir)
    raise ValueError(f"不支持的报告格式: {report_format}")
