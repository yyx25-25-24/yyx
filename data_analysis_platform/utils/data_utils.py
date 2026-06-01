import pandas as pd
import numpy as np
from typing import List, Tuple


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    # 将数值类型视作数值列，同时将高基数的整数字符串尝试转换
    nums = []
    for c in df.columns:
        try:
            if df[c].dtype.kind in "fi":
                nums.append(c)
            else:
                # 低比例缺失且全部可转为数值的列也当作数值列
                ser = pd.to_numeric(df[c], errors="coerce")
                nonnull = ser.notna().sum()
                if nonnull / max(1, len(df)) > 0.8 and ser.nunique() > 5:
                    nums.append(c)
        except Exception:
            continue
    return nums


def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    cats = []
    for c in df.columns:
        try:
            if df[c].dtype.kind not in "fi":
                cats.append(c)
            else:
                # 如果数值列基数很小，视作类别（例如等级编码）
                if df[c].nunique() <= 15:
                    cats.append(c)
        except Exception:
            continue
    return cats


def detect_time_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    for c in df.columns:
        try:
            ser = pd.to_datetime(df[c], errors="coerce")
            nonnull = ser.notna().sum()
            if nonnull > 0 and nonnull / len(df) > 0.5:
                candidates.append(c)
        except Exception:
            continue
    return candidates


def preprocess_df(df: pd.DataFrame, parse_dates: bool = True, fill_method: str = "ffill", normalize: bool = False) -> pd.DataFrame:
    df = df.copy()
    if parse_dates:
        time_cols = detect_time_columns(df)
        for c in time_cols:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass

    if fill_method and fill_method in ("ffill", "bfill"):
        df = df.fillna(method=fill_method)
    else:
        # fill numeric with median
        num_cols = detect_numeric_columns(df)
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())

    if normalize:
        num_cols = detect_numeric_columns(df)
        for c in num_cols:
            col = df[c].astype(float)
            if col.std() != 0:
                df[c] = (col - col.mean()) / col.std()
    # 尝试类型转换：将可以安全转换为数值的列转为数值类型
    for c in df.columns:
        try:
            if df[c].dtype == object:
                coerced = pd.to_numeric(df[c], errors='coerce')
                if coerced.notna().sum() / max(1, len(df)) > 0.8:
                    df[c] = coerced
        except Exception:
            continue
    return df


def choose_chart_columns(df: pd.DataFrame, intent: str) -> Tuple[str, str]:
    """Return (x_col, y_col) best suited for the intent.
    Simple heuristic: prefer time column for x, else categorical, else index.
    """
    num_cols = detect_numeric_columns(df)
    time_cols = detect_time_columns(df)
    cat_cols = detect_categorical_columns(df)

    y_col = num_cols[0] if num_cols else (df.columns[0] if len(df.columns) else None)
    if intent and ("scatter" in intent or "相关" in intent):
        if len(num_cols) >= 2:
            return num_cols[0], num_cols[1]
    if time_cols:
        x_col = time_cols[0]
    elif cat_cols:
        x_col = cat_cols[0]
    else:
        x_col = df.index.name or df.index
    return x_col, y_col
