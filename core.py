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


def read_csv_series(file) -> Tuple[np.ndarray, Optional[pd.Series]]:
    df = pd.read_csv(file)
    if df.empty:
        raise ValueError("CSV 为空")

    lower_cols = {c.lower(): c for c in df.columns}
    date_col = lower_cols.get("date") or lower_cols.get("ds")
    value_col = lower_cols.get("value") or lower_cols.get("y")

    if date_col and value_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        values = pd.to_numeric(df[value_col], errors="coerce")
        mask = dates.notna() & values.notna()
        dates = dates[mask]
        values = values[mask]
        if values.empty:
            raise ValueError("未能从 CSV 中解析出有效数值")
        order = np.argsort(dates.to_numpy())
        return values.to_numpy(dtype=float)[order], dates.iloc[order]

    if df.shape[1] < 2:
        raise ValueError("CSV 至少需要两列（日期 + 数值）")

    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    mask = dates.notna() & values.notna()
    dates = dates[mask]
    values = values[mask]
    if values.empty:
        raise ValueError("未能从 CSV 的前两列解析出有效数值")
    order = np.argsort(dates.to_numpy())
    return values.to_numpy(dtype=float)[order], dates.iloc[order]


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

