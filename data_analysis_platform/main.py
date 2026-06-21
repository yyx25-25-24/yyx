import os
import pandas as pd
from datetime import datetime
import config
from utils import session_store
from utils.chart_utils import detect_chart_type, generate_default_chart
from utils.report_utils import save_analysis_report
from utils.data_utils import preprocess_df, choose_chart_columns
from utils.llm_utils import LLMClient
from utils.tool_registry import registry, ToolRegistry
from utils.retry_engine import RetryEngine, RetryConfig
from utils.result_validator import validator, ValidationResult
all_data = {}
current_df = None

retry_engine = RetryEngine(
    default_config=RetryConfig(
        max_retries=config.REQUEST_RETRIES,
        base_delay=config.REQUEST_BACKOFF,
        backoff_factor=2.0
    )
)


def init_platform():
    """初始化平台：读取所有Excel文件，创建输出目录"""
    print("=" * 70)
    print("🚀 通用型大模型自动化数据分析平台 启动中...")
    print("=" * 70)
    
    os.makedirs(config.CHART_SAVE_PATH, exist_ok=True)
    os.makedirs(config.REPORT_SAVE_PATH, exist_ok=True)
    os.makedirs(config.LOG_SAVE_PATH, exist_ok=True)
    
    print(f"📁 输出文件夹已创建：{config.CHART_SAVE_PATH}、{config.REPORT_SAVE_PATH}、{config.LOG_SAVE_PATH}")
    
    if not config.DASHSCOPE_API_KEY:
        print("⚠️ 警告：未配置 DASHSCOPE_API_KEY，将启用本地分析模式（无需大模型）")
    
    if not config.DATA_FILES:
        print("❌ 错误：DATA_FILES 列表为空，请在 config.py 中配置数据文件路径")
        raise SystemExit(1)
    
    print("\n📊 正在读取数据文件...")
    for file_path in config.DATA_FILES:
        if not os.path.exists(file_path):
            print(f"⚠️  警告：文件 {file_path} 不存在，跳过")
            continue
        
        excel_file = None
        for engine in ['openpyxl', 'xlrd']:
            try:
                excel_file = pd.ExcelFile(file_path, engine=engine)
                print(f"✅ 使用 {engine} 引擎成功读取文件：{os.path.basename(file_path)}")
                break
            except Exception as e:
                print(f"⚠️  使用 {engine} 引擎读取失败：{str(e)}")
                continue
        
        if excel_file is None:
            print(f"❌ 所有引擎都无法读取文件 {file_path}，跳过")
            continue
        
        for sheet_name in excel_file.sheet_names:
            try:
                df = excel_file.parse(sheet_name)
                if df.empty:
                    print(f"⚠️  跳过空sheet：{os.path.basename(file_path)}_{sheet_name}")
                    continue
                
                data_key = f"{os.path.basename(file_path).split('.')[0]}_{sheet_name}"
                all_data[data_key] = df
                print(f"✅ 已加载：{data_key} | {len(df)} 行 × {len(df.columns)} 列")
            except Exception as sheet_error:
                print(f"⚠️  读取sheet {sheet_name} 失败：{sheet_error}，跳过")
    
    if not all_data:
        print("\n❌ 错误：未加载到任何有效数据")
        print("💡 解决方法：")
        print("1. 打开你的Excel文件，另存为 .xlsx 格式")
        print("2. 确保文件没有损坏，可以正常打开")
        print("3. 检查config.py中的文件路径是否正确")
        raise SystemExit(1)
    
    global current_df
    current_df = list(all_data.values())[0]
    try:
        current_df = preprocess_df(current_df, parse_dates=True, fill_method="ffill")
        all_data[list(all_data.keys())[0]] = current_df
    except Exception:
        pass
    print(f"\n📌 默认数据集已设置：{list(all_data.keys())[0]}")
    print(f"📋 数据列名：{list(current_df.columns)}")
    print("\n" + "=" * 70)
    print("✅ 平台初始化完成！")
    print("=" * 70)


@registry.register(
    name="dataframe_analysis",
    description="基于当前数据集执行本地数据分析",
    category="analysis",
    requires_df=True,
    tags=["analysis", "local"]
)
def dataframe_analysis_tool(query: str, df: pd.DataFrame) -> str:
    """本地数据分析引擎"""
    return local_analysis_engine(query, df)


@registry.register(
    name="chart_generation",
    description="生成数据可视化图表",
    category="visualization",
    requires_df=True,
    tags=["chart", "visualization"]
)
def chart_generation_tool(df: pd.DataFrame, chart_type: str, title: str = None) -> str:
    """图表生成工具"""
    try:
        chart_path = generate_default_chart(
            df, 
            chart_type, 
            config.CHART_SAVE_PATH, 
            config.CHART_FORMAT,
            title=title or f"自动生成图表 - {chart_type}"
        )
        return f"✅ 图表已保存：{chart_path}"
    except Exception as e:
        return f"❌ 图表生成失败：{str(e)}"


@registry.register(
    name="report_export",
    description="导出分析报告",
    category="export",
    requires_df=False,
    tags=["report", "export"]
)
def report_export_tool(query: str, result: str, dataset_key: str) -> str:
    """报告导出工具"""
    try:
        report_path = save_analysis_report(
            query, 
            result, 
            dataset_key, 
            config.REPORT_SAVE_PATH, 
            config.REPORT_FORMAT
        )
        return f"✅ 报告已保存：{report_path}"
    except Exception as e:
        return f"❌ 报告导出失败：{str(e)}"


def create_analysis_agent(df):
    """创建大模型分析 Agent，集成工具注册中心"""
    try:
        from langchain import agents
        from langchain.tools import tool
        from langchain_community.llms import Tongyi
    except Exception as exc:
        try:
            client = LLMClient(
                backend=config.LLM_BACKEND, 
                openai_api_key=config.OPENAI_API_KEY, 
                dashscope_key=config.DASHSCOPE_API_KEY, 
                model_name=config.MODEL_NAME, 
                temperature=config.TEMPERATURE, 
                retries=config.REQUEST_RETRIES, 
                backoff=config.REQUEST_BACKOFF
            )
            class SimpleAgent:
                def __init__(self, client):
                    self.client = client
                def run(self, prompt: str):
                    return self.client.send(prompt)
            return SimpleAgent(client)
        except Exception:
            print("⚠️ 大模型依赖未就绪，将启用本地分析模式。详细原因：", exc)
            return None

    try:
        llm = Tongyi(
            model_name=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            api_key=config.DASHSCOPE_API_KEY
        )
    except Exception as exc:
        try:
            client = LLMClient(
                backend=config.LLM_BACKEND, 
                openai_api_key=config.OPENAI_API_KEY, 
                dashscope_key=config.DASHSCOPE_API_KEY, 
                model_name=config.MODEL_NAME, 
                temperature=config.TEMPERATURE, 
                retries=config.REQUEST_RETRIES, 
                backoff=config.REQUEST_BACKOFF
            )
            class SimpleAgent:
                def __init__(self, client):
                    self.client = client
                def run(self, prompt: str):
                    return self.client.send(prompt)
            return SimpleAgent(client)
        except Exception:
            print("⚠️ Tongyi 初始化失败，将启用本地分析模式。详细原因：", exc)
            return None

    def wrapped_analysis_tool(query: str) -> str:
        """包装后的分析工具，集成重试和校验"""
        def execute_analysis():
            result = dataframe_analysis_tool(query, df)
            
            validation = validator.validate(result, "text")
            if not validation.is_valid:
                raise ValueError(f"结果校验失败: {'; '.join(validation.errors)}")
            
            if validation.warnings:
                print(f"⚠️ 结果警告: {'; '.join(validation.warnings)}")
            
            return validation.sanitized_output
        
        try:
            return retry_engine.execute_with_retry(
                execute_analysis,
                config=RetryConfig(
                    max_retries=2,
                    base_delay=1.0,
                    retryable_exceptions=(ValueError, RuntimeError)
                )
            )
        except Exception as e:
            print(f"⚠️ 分析执行失败，返回降级结果: {e}")
            return f"分析执行失败: {str(e)}"

    try:
        tool_instance = tool(wrapped_analysis_tool)
        agent = agents.create_agent(
            model=llm,
            tools=[tool_instance],
            system_prompt=(
                "你是一个专业的数据分析专家，严格使用提供的工具回答用户问题，"
                "并给出简洁、结构化的中文分析结论。"
            )
        )
        return agent
    except Exception as exc:
        print("⚠️ 大模型 Agent 创建失败，将启用本地分析模式。详细原因：", exc)
        return None


def local_analysis_engine(query: str, df: pd.DataFrame) -> str:
    """本地数据分析引擎，用于无大模型时的回退分析。"""
    text = query.lower()
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
    if not numeric_cols:
        return "当前数据集中没有可用于分析的数值列，请更换数据集或检查数据文件。"

    if "相关" in text or "相关性" in text:
        return summarize_correlation(df, numeric_cols)
    if "趋势" in text or "变化" in text or "走势" in text:
        return summarize_trend(df, numeric_cols)
    if "前" in text or "排名" in text or "top" in text or "最大" in text or "最小" in text:
        return summarize_top_items(df, numeric_cols)
    if "列名" in text or "字段" in text or "数据集" in text:
        return summarize_columns(df)
    return summarize_overall(df, numeric_cols)


def summarize_columns(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    return (
        "当前数据集可用列如下：\n" +
        "\n".join([f"- {col} ({df[col].dtype})" for col in cols]) +
        "\n\n你可以输入更具体的分析需求，例如分析趋势、排名或相关性。"
    )


def summarize_overall(df: pd.DataFrame, numeric_cols: list[str]) -> str:
    desc = df[numeric_cols].describe().round(3).to_string()
    return (
        '当前数据集数值列摘要统计如下：\n' +
        desc +
        '\n\n提示：你可以输入"趋势""排名""相关性"等关键词，让平台生成更具体的分析结果。'
    )


def summarize_trend(df: pd.DataFrame, numeric_cols: list[str]) -> str:
    col = numeric_cols[0]
    values = df[col].dropna()
    if len(values) < 2:
        return f"数值列 {col} 数据不足，无法分析趋势。"
    first, last = values.iloc[0], values.iloc[-1]
    direction = "上升" if last > first else "下降" if last < first else "持平"
    return (
        f"建议分析列：{col}。\n"
        f"该列首尾值分别为 {first} 和 {last}，整体走势为：{direction}。\n"
        f"如果你想要更准确的趋势分析，可以指定具体的列名和时间序列列。"
    )


def summarize_top_items(df: pd.DataFrame, numeric_cols: list[str]) -> str:
    num_col = numeric_cols[0]
    cat_cols = [c for c in df.columns if df[c].dtype.kind not in "fi"]
    if cat_cols:
        group_col = cat_cols[0]
        sorted_df = df.sort_values(by=num_col, ascending=False).head(5)
        items = sorted_df[[group_col, num_col]].dropna().to_dict(orient="records")
        lines = [f"{item[group_col]}：{item[num_col]}" for item in items]
        return (
            f"按 {num_col} 排名前 5 的项为：\n" +
            "\n".join(lines)
        )
    top_items = df.sort_values(by=num_col, ascending=False).head(5)[num_col].tolist()
    return f"按 {num_col} 排名前 5 的值为：{top_items}。"


def summarize_correlation(df: pd.DataFrame, numeric_cols: list[str]) -> str:
    corr = df[numeric_cols].corr().round(3)
    return (
        "数值列相关性矩阵：\n" +
        corr.to_string() +
        "\n\n提示：你可以进一步指定要分析的两个列名。"
    )


def _find_dataset_key(dataset_key: str) -> str | None:
    if dataset_key in all_data:
        return dataset_key
    candidates = [k for k in all_data if dataset_key.lower() in k.lower()]
    return candidates[0] if len(candidates) == 1 else None


def switch_dataset(dataset_key):
    """切换当前分析的数据集"""
    global current_df
    matched_key = _find_dataset_key(dataset_key)
    if matched_key is None:
        print(f"❌ 错误：未找到数据集 {dataset_key}")
        print(f"📋 可用数据集：{list(all_data.keys())}")
        return False
    current_df = all_data[matched_key]
    print(f"✅ 已切换到数据集：{matched_key}")
    print(f"📋 数据列名：{list(current_df.columns)}")
    return True


def show_help():
    """显示帮助信息"""
    print("\n" + "=" * 70)
    print("📖 平台使用帮助")
    print("=" * 70)
    print("1. 基础分析：直接输入自然语言分析需求，例如：")
    print("   - 分析2012-2023年氮磷排放的总体变化趋势，生成折线图")
    print("   - 找出2023年碳排放最高的5个区县，生成柱状图")
    print("   - 分析数字化指数与氮磷排放的相关性，生成散点图")
    print("\n2. 切换数据集：输入 switch 数据集名称，例如：")
    print(f"   - switch {list(all_data.keys())[1] if len(all_data) > 1 else '数据集名称'}")
    print(f"📋 可用数据集：{list(all_data.keys())}")
    print("\n3. 查看已注册工具：输入 tools")
    print("\n4. 其他命令：")
    print("   - help：显示帮助信息")
    print("   - exit/退出/q：退出程序")
    print("=" * 70 + "\n")


def show_registered_tools():
    """显示已注册的工具列表"""
    print("\n" + "=" * 70)
    print("🔧 已注册的工具")
    print("=" * 70)
    tools = registry.list_tools()
    for tool in tools:
        print(f"📦 [{tool.category}] {tool.name}")
        print(f"   描述: {tool.description}")
        print(f"   标签: {', '.join(tool.tags)}")
        print()
    print("=" * 70 + "\n")


def main():
    init_platform()
    show_help()
    agent = create_analysis_agent(current_df)
    if agent is None:
        print("⚠️ 当前未启用大模型分析；平台将使用本地数据分析引擎。")

    while True:
        user_input = input("请输入分析需求/命令：").strip()
        if user_input.lower() in ["exit", "退出", "q"]:
            print("\n👋 平台已退出，感谢使用！")
            break
        if user_input.lower() == "help":
            show_help()
            continue
        if user_input.lower() == "tools":
            show_registered_tools()
            continue
        if user_input.lower() == "batch":
            print("\n🔁 开始批处理：对所有数据集运行预设分析并导出报告与图表...")
            run_batch(agent)
            print("🔁 批处理完成")
            continue
        if user_input.lower() == "list":
            print(f"📋 可用数据集：{list(all_data.keys())}")
            continue
        if user_input.lower() == "current":
            print(f"📌 当前数据集：{next((k for k, v in all_data.items() if v is current_df), '未知')}\n")
            print(f"📋 当前列名：{list(current_df.columns)}")
            continue
        if user_input.lower().startswith("switch"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("❌ 错误：请指定要切换的数据集名称，例如：switch 17县原始数据_Sheet1")
                continue
            dataset_key = parts[1].strip()
            if switch_dataset(dataset_key):
                agent = create_analysis_agent(current_df)
            continue
        if not user_input:
            print("⚠️  请输入有效的分析需求")
            continue

        print("\n🔍 正在分析数据，请稍候...")
        try:
            result = None
            if agent is not None:
                try:
                    full_query = f"""
                    你是一个专业的数据分析专家，严格按照以下要求执行：
                    1. 所有分析必须基于提供的数据集，不得编造数据
                    2. 如果需要可视化，优先生成适合的图表并保存到 {config.CHART_SAVE_PATH}
                    3. 分析报告必须结构化，包含数据结论、图表说明和业务建议
                    4. 所有回答必须使用中文，专业、简洁、逻辑清晰
                    5. 生成的代码必须安全可运行，不得有危险操作

                    用户的分析需求：{user_input}
                    """
                    result = agent.run(full_query)
                except Exception as llm_error:
                    print("⚠️ 大模型分析执行失败，将回退到本地分析。详细原因：", llm_error)
                    result = local_analysis_engine(user_input, current_df)
            else:
                result = local_analysis_engine(user_input, current_df)

            validation = validator.validate(result, "text")
            if not validation.is_valid:
                print(f"⚠️ 结果校验警告: {'; '.join(validation.errors)}")
                result = validation.sanitized_output or result
            
            print("\n" + "-" * 70)
            print("📊 分析结果")
            print("-" * 70)
            print(result)
            print("-" * 70 + "\n")

            dataset_key = next((k for k, v in all_data.items() if v is current_df), "unknown_dataset")
            report_path = save_analysis_report(user_input, result, dataset_key, config.REPORT_SAVE_PATH, config.REPORT_FORMAT)
            print(f"✅ 分析报告已保存：{report_path}")

            chart_type = detect_chart_type(user_input)
            try:
                chart_path = generate_default_chart(
                    current_df,
                    chart_type,
                    config.CHART_SAVE_PATH,
                    config.CHART_FORMAT,
                    title=f"{dataset_key} - {chart_type} 图"
                )
                print(f"✅ 自动图表已保存：{chart_path}")
            except Exception as chart_error:
                print(f"⚠️ 未生成图表：{chart_error}")

            charts = [chart_path] if 'chart_path' in locals() else None
            try:
                if charts:
                    save_analysis_report(user_input, (result, charts), dataset_key, config.REPORT_SAVE_PATH, config.REPORT_FORMAT)
            except Exception:
                pass

            try:
                session_store.append_session({
                    "dataset": dataset_key,
                    "query": user_input,
                    "result_preview": str(result)[:200],
                    "charts": charts or [],
                })
            except Exception:
                pass

            _save_query_log(dataset_key, user_input, result)
        except Exception as e:
            print(f"\n❌ 分析出错：{str(e)}")
            print("💡 建议：重新描述你的需求，确保需求清晰明确\n")


def _save_query_log(dataset_key: str, query: str, result: str) -> None:
    os.makedirs(config.LOG_SAVE_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interaction_{dataset_key}_{timestamp}.log"
    filename = filename.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(config.LOG_SAVE_PATH, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集: {dataset_key}\n")
        f.write(f"查询: {query}\n\n")
        f.write("分析结果:\n")
        f.write(result)
        f.write("\n")


if __name__ == "__main__":
    main()


def run_batch(agent):
    """对所有已加载数据集运行一组预设分析并导出报告与图表"""
    preset_queries = [
        "分析总体趋势，生成折线图",
        "找出数值列中排名前5的项，生成柱状图",
        "计算数值列之间的相关性，生成相关性矩阵"
    ]
    for dataset_key, df in list(all_data.items()):
        print(f"\n➡️ 批处理数据集：{dataset_key}")
        try:
            df_proc = preprocess_df(df, parse_dates=True, fill_method="ffill")
            all_data[dataset_key] = df_proc
        except Exception:
            df_proc = df

        for q in preset_queries:
            print(f"  - 查询: {q}")
            if agent is not None:
                try:
                    resp = agent.run(q)
                except Exception as e:
                    print(f"    ⚠️ 大模型执行失败：{e}，使用本地分析回退")
                    resp = local_analysis_engine(q, df_proc)
            else:
                resp = local_analysis_engine(q, df_proc)

            chart_paths = []
            chart_type = detect_chart_type(q)
            try:
                chart_path = generate_default_chart(df_proc, chart_type, config.CHART_SAVE_PATH, config.CHART_FORMAT, title=f"{dataset_key} - {q}")
                chart_paths.append(chart_path)
                print(f"    ✅ 图表: {chart_path}")
            except Exception as chart_err:
                print(f"    ⚠️ 未生成图表：{chart_err}")

            report_path = save_analysis_report(q, (resp, chart_paths) if chart_paths else resp, dataset_key, config.REPORT_SAVE_PATH, config.REPORT_FORMAT)
            print(f"    ✅ 报告: {report_path}")

            try:
                session_store.append_session({
                    "dataset": dataset_key,
                    "query": q,
                    "result_preview": str(resp)[:200],
                    "charts": chart_paths,
                })
            except Exception:
                pass

            chart_type = detect_chart_type(q)
            try:
                x_col, y_col = choose_chart_columns(df_proc, q)
                chart_path = generate_default_chart(df_proc, chart_type, config.CHART_SAVE_PATH, config.CHART_FORMAT, title=f"{dataset_key} - {q}")
                print(f"    ✅ 图表: {chart_path}")
            except Exception as chart_err:
                print(f"    ⚠️ 未生成图表：{chart_err}")