from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


Granularity = Literal["daily", "weekly"]


@dataclass(frozen=True)
class ForecastResult:
    granularity: Granularity
    history: np.ndarray
    fitted: np.ndarray
    future: np.ndarray
    future_total: float
    r2: float
    mae: float
    baseline_future_total: float
    trend_slope: float
    trend_intercept: float
    boot_low: Optional[float]
    boot_high: Optional[float]


@dataclass(frozen=True)
class ParsedInput:
    values: np.ndarray
    dates: Optional[pd.Series]


@dataclass(frozen=True)
class CsvParseMeta:
    value_col: str
    date_col: Optional[str]


def parse_numeric_series(text: str) -> np.ndarray:
    cleaned = text.replace("，", ",").replace("；", ",")
    parts = re.split(r"[\s,]+", cleaned.strip())
    values: list[float] = []
    for p in parts:
        if not p:
            continue
        values.append(float(p))
    return np.asarray(values, dtype=float)


def infer_granularity_from_dates(dates: pd.Series) -> Granularity:
    sorted_dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(sorted_dates) < 3:
        return "daily"
    diffs = sorted_dates.diff().dropna().dt.total_seconds() / (24 * 3600)
    median_days = float(diffs.median())
    return "daily" if median_days <= 2.0 else "weekly"


def read_csv_series(file) -> Tuple[np.ndarray, Optional[pd.Series], CsvParseMeta]:
    import io
    from pandas.errors import EmptyDataError

    content: bytes | None = None
    if isinstance(file, (bytes, bytearray)):
        content = bytes(file)
    elif hasattr(file, "getvalue"):
        try:
            content = file.getvalue()
        except Exception:
            content = None
    if content is None and hasattr(file, "read"):
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            content = file.read()
        except Exception:
            content = None

    try:
        if content is not None:
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", sep=None, engine="python")
        else:
            df = pd.read_csv(file, encoding="utf-8-sig", sep=None, engine="python")
    except EmptyDataError:
        raise ValueError("CSV 无法解析：文件为空或格式不正确（请确认是标准 CSV，而不是空文件/加密文件/错误分隔符）")
    if df.empty:
        raise ValueError("CSV 为空")

    df = df.dropna(axis=1, how="all")
    if df.empty:
        raise ValueError("CSV 为空")

    def _name_score(col: str) -> float:
        s = str(col).strip().lower()
        score = 0.0
        for kw in [
            "value",
            "y",
            "sales",
            "traffic",
            "count",
            "num",
            "people",
            "visitor",
            "visitors",
            "uv",
            "pv",
            "客流",
            "人流",
            "人数",
            "销量",
            "销售",
            "订单",
            "访客",
        ]:
            if kw in s:
                score += 1.2
        for bad in ["id", "uuid", "code", "编号", "代码", "店id", "merchantid", "shopid"]:
            if bad in s:
                score -= 1.5
        return score

    def _date_score(col: str) -> float:
        s = str(col).strip().lower()
        if s in {"date", "ds", "day", "time", "datetime", "日期", "时间"}:
            return 2.0
        if any(k in s for k in ["date", "day", "time", "日期", "时间", "周", "month", "year"]):
            return 1.0
        return 0.0

    n = len(df)
    date_candidates: list[tuple[float, str, pd.Series]] = []
    numeric_candidates: list[tuple[float, str, pd.Series]] = []

    for col in df.columns:
        s_date = pd.to_datetime(df[col], errors="coerce")
        valid_date = int(s_date.notna().sum())
        if valid_date >= max(3, int(0.6 * n)):
            date_candidates.append((valid_date + 10.0 * _date_score(col), str(col), s_date))

        s_num = pd.to_numeric(df[col], errors="coerce")
        valid_num = int(s_num.notna().sum())
        if valid_num >= max(5, int(0.6 * n)):
            std = float(np.nanstd(s_num.to_numpy(dtype=float)))
            numeric_candidates.append(
                (valid_num + 0.05 * std + 10.0 * _name_score(col), str(col), s_num)
            )

    if not numeric_candidates:
        if df.shape[1] == 1:
            col = str(df.columns[0])
            s_num = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            s_num = s_num.dropna()
            if s_num.empty:
                raise ValueError("无法在 CSV 中找到数值列：请确保至少有一列是数字")
            return s_num.to_numpy(dtype=float), None, CsvParseMeta(value_col=col, date_col=None)
        raise ValueError("无法在 CSV 中找到数值列：请确保至少有一列是数字")

    numeric_candidates.sort(key=lambda x: x[0], reverse=True)
    value_col, values_s = numeric_candidates[0][1], numeric_candidates[0][2]

    date_col: Optional[str] = None
    dates_s: Optional[pd.Series] = None
    if date_candidates:
        date_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, cand_col, cand_series in date_candidates:
            if cand_col != value_col:
                date_col = cand_col
                dates_s = cand_series
                break

    if dates_s is not None:
        mask = dates_s.notna() & values_s.notna()
        dates_s = dates_s[mask]
        values_s = values_s[mask]
        if values_s.empty:
            raise ValueError("未能从 CSV 中解析出有效数值")
        dates_s = pd.to_datetime(dates_s, errors="coerce")
        mask2 = dates_s.notna() & values_s.notna()
        dates_s = dates_s[mask2]
        values_s = values_s[mask2]
        if values_s.empty:
            raise ValueError("未能从 CSV 中解析出有效数值")
        order = np.argsort(dates_s.to_numpy())
        return (
            values_s.to_numpy(dtype=float)[order],
            pd.Series(dates_s.iloc[order]),
            CsvParseMeta(value_col=value_col, date_col=date_col),
        )

    values_s = values_s.dropna()
    if values_s.empty:
        raise ValueError("未能从 CSV 中解析出有效数值")
    return values_s.to_numpy(dtype=float), None, CsvParseMeta(value_col=value_col, date_col=None)


def make_template_csv(granularity: Granularity) -> bytes:
    today = date.today()
    if granularity == "daily":
        n = 30
        dates = [today - timedelta(days=n - 1 - i) for i in range(n)]
        values = np.linspace(120, 135, n).round(0).astype(int)
    else:
        n = 16
        dates = [today - timedelta(days=7 * (n - 1 - i)) for i in range(n)]
        values = np.linspace(900, 1020, n).round(0).astype(int)
    df = pd.DataFrame({"date": pd.to_datetime(dates), "value": values})
    return df.to_csv(index=False).encode("utf-8")


def fit_forecast(
    values: np.ndarray,
    granularity: Granularity,
    holdout_points: int,
    bootstrap_trials: int,
    bootstrap_alpha: float,
) -> ForecastResult:
    if values.ndim != 1:
        values = values.reshape(-1)
    values = np.asarray(values, dtype=float)
    if len(values) < max(14, holdout_points + 8):
        raise ValueError("数据点太少：请至少提供 20 个以上数值")

    if np.any(~np.isfinite(values)):
        raise ValueError("数据包含无效值（NaN/inf）")

    values = np.clip(values, 0.0, None)

    n = len(values)
    x = np.arange(n, dtype=float).reshape(-1, 1)
    y = values

    holdout_points = int(np.clip(holdout_points, 7, max(7, n - 7)))
    split = n - holdout_points
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred_test = model.predict(x_test)
    r2 = float(r2_score(y_test, y_pred_test)) if len(y_test) >= 2 else float("nan")
    mae = float(mean_absolute_error(y_test, y_pred_test))

    fitted = model.predict(x)

    future_steps = 7 if granularity == "daily" else 1
    x_future = np.arange(n, n + future_steps, dtype=float).reshape(-1, 1)
    future = np.clip(model.predict(x_future), 0.0, None)
    future_total = float(np.sum(future)) if granularity == "daily" else float(future[0])

    if granularity == "daily":
        baseline_future_total = float(np.sum(y[-7:]))
    else:
        baseline_future_total = float(y[-1])

    residuals = y_test - y_pred_test
    boot_low: Optional[float]
    boot_high: Optional[float]

    if bootstrap_trials >= 200 and len(residuals) >= 8:
        rng = np.random.default_rng(7)
        residuals = residuals.astype(float)
        samples = rng.choice(residuals, size=(bootstrap_trials, future_steps), replace=True)
        future_matrix = np.clip(future.reshape(1, -1) + samples, 0.0, None)
        totals = (
            np.sum(future_matrix, axis=1)
            if granularity == "daily"
            else future_matrix[:, 0]
        )
        low_q = float(np.quantile(totals, bootstrap_alpha / 2))
        high_q = float(np.quantile(totals, 1 - bootstrap_alpha / 2))
        boot_low = max(0.0, low_q)
        boot_high = max(0.0, high_q)
    else:
        boot_low = None
        boot_high = None

    return ForecastResult(
        granularity=granularity,
        history=y,
        fitted=fitted,
        future=future,
        future_total=future_total,
        r2=r2,
        mae=mae,
        baseline_future_total=baseline_future_total,
        trend_slope=float(model.coef_[0]),
        trend_intercept=float(model.intercept_),
        boot_low=boot_low,
        boot_high=boot_high,
    )


def format_number(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.1f}"

