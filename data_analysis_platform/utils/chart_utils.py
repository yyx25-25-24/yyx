import os
import datetime
import plotly.express as px
import plotly.io as pio


def detect_chart_type(user_input: str) -> str:
    text = user_input.lower()
    if "箱" in text or "箱型" in text:
        return "box"
    if "热力" in text or "热图" in text:
        return "heatmap"
    if "散点" in text or "相关" in text:
        return "scatter"
    if "折线" in text or "趋势" in text:
        return "line"
    if "堆叠" in text or "堆叠柱" in text:
        return "stacked_bar"
    if "柱状" in text or "柱形" in text:
        return "bar"
    return "line"


def _get_numeric_columns(df):
    return [c for c in df.columns if df[c].dtype.kind in "fi"]


def _get_categorical_columns(df):
    return [c for c in df.columns if df[c].dtype.kind not in "fi"]


def generate_default_chart(df, chart_type: str, save_dir: str, chart_format: str, title: str = None, x_col: str = None, y_col: str = None) -> str:
    os.makedirs(save_dir, exist_ok=True)
    numeric_cols = _get_numeric_columns(df)
    if not numeric_cols:
        raise ValueError("当前数据集中没有可用于绘图的数值列")

    chart_type = chart_type.lower()
    title = title or f"自动生成图表 - {chart_type}"

    chart_type = chart_type.lower()
    # allow overriding columns
    if x_col is None or y_col is None:
        # best-effort selection
        time_cols = [c for c in df.columns if str(df[c].dtype).startswith('datetime') or 'datetime' in str(df[c].dtype)]
        cat_cols = _get_categorical_columns(df)
        if chart_type in ("scatter", "heatmap"):
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
            else:
                x_col, y_col = numeric_cols[0], numeric_cols[0]
        elif chart_type in ("bar", "stacked_bar"):
            if cat_cols:
                x_col = cat_cols[0]
            else:
                df = df.reset_index()
                x_col = df.columns[0]
            y_col = numeric_cols[0]
        elif chart_type == "box":
            # box plot by first numeric, grouped by first categorical if exists
            y_col = numeric_cols[0]
            x_col = cat_cols[0] if cat_cols else None
        elif chart_type == "heatmap":
            # heatmap of correlation
            corr = df[numeric_cols].corr()
            import plotly.figure_factory as ff
            fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Viridis')
        else:
            # line default
            if time_cols:
                x_col = time_cols[0]
            else:
                df = df.reset_index()
                x_col = df.columns[0]
            y_col = numeric_cols[0]

    # build figure if not already created by heatmap
    if 'fig' not in locals():
        if chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        elif chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        elif chart_type == "stacked_bar":
            # attempt stacked bar by second numeric as stack if available
            if len(numeric_cols) >= 2 and x_col is not None:
                fig = px.bar(df, x=x_col, y=[y_col, numeric_cols[1]], title=title)
            else:
                fig = px.bar(df, x=x_col, y=y_col, title=title)
        elif chart_type == "box":
            if x_col:
                fig = px.box(df, x=x_col, y=y_col, title=title)
            else:
                fig = px.box(df, y=y_col, title=title)
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chart_{chart_type}_{timestamp}.{chart_format}"
    save_path = os.path.join(save_dir, filename)

    if chart_format == "html":
        pio.write_html(fig, save_path, auto_open=False)
    else:
        try:
            pio.write_image(fig, save_path)
        except Exception as exc:
            raise RuntimeError(
                f"无法导出为 {chart_format}，请确认已安装 kaleido 或使用 html 格式。详细错误: {exc}"
            )
    return save_path
