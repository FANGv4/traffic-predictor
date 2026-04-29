from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
except ModuleNotFoundError:
    st.set_page_config(page_title="客流/销售预测器", page_icon="📈", layout="wide")
    st.error("缺少依赖：未安装 scikit-learn。请先运行：pip install -r requirements.txt")
    st.stop()


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
    approx_low: Optional[float]
    approx_high: Optional[float]


@dataclass(frozen=True)
class ParsedInput:
    values: np.ndarray
    dates: Optional[pd.Series]


def _parse_numeric_series(text: str) -> np.ndarray:
    cleaned = text.replace("，", ",").replace("；", ",")
    parts = re.split(r"[\s,]+", cleaned.strip())
    values: list[float] = []
    for p in parts:
        if not p:
            continue
        values.append(float(p))
    return np.asarray(values, dtype=float)


def _infer_granularity_from_dates(dates: pd.Series) -> Granularity:
    sorted_dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(sorted_dates) < 3:
        return "daily"
    diffs = sorted_dates.diff().dropna().dt.total_seconds() / (24 * 3600)
    median_days = float(diffs.median())
    return "daily" if median_days <= 2.0 else "weekly"


def _read_csv_series(file) -> Tuple[np.ndarray, Optional[pd.Series]]:
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


def _fit_forecast(values: np.ndarray, granularity: Granularity) -> ForecastResult:
    if values.ndim != 1:
        values = values.reshape(-1)
    values = np.asarray(values, dtype=float)
    if len(values) < 14:
        raise ValueError("数据点太少：至少需要 14 个数值")

    if np.any(~np.isfinite(values)):
        raise ValueError("数据包含无效值（NaN/inf）")

    values = np.clip(values, 0.0, None)

    n = len(values)
    x = np.arange(n, dtype=float).reshape(-1, 1)
    y = values

    split = max(int(n * 0.8), n - 14)
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

    residuals = y_test - y_pred_test
    if len(residuals) >= 5:
        sigma = float(np.std(residuals, ddof=1))
        approx_low = max(0.0, future_total - 1.28 * sigma * (np.sqrt(future_steps)))
        approx_high = max(0.0, future_total + 1.28 * sigma * (np.sqrt(future_steps)))
    else:
        approx_low = None
        approx_high = None

    return ForecastResult(
        granularity=granularity,
        history=y,
        fitted=fitted,
        future=future,
        future_total=future_total,
        r2=r2,
        mae=mae,
        approx_low=approx_low,
        approx_high=approx_high,
    )


def _format_number(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.1f}"


st.set_page_config(
    page_title="客流/销售预测器",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
<style>
  .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
  [data-testid="stMetricValue"] { font-size: 1.65rem; }
  .app-hero { border: 1px solid rgba(15, 23, 42, 0.08); background: linear-gradient(180deg, rgba(46,107,230,0.08), rgba(255,255,255,0)); padding: 16px 18px; border-radius: 14px; }
  .app-card { border: 1px solid rgba(15, 23, 42, 0.08); background: #ffffff; padding: 14px 16px; border-radius: 14px; }
  .app-muted { color: rgba(15, 23, 42, 0.72); }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="app-hero">
  <div style="display:flex; justify-content:space-between; align-items:flex-end; gap: 12px; flex-wrap: wrap;">
    <div>
      <div style="font-size: 22px; font-weight: 700;">客流 / 销售预测器</div>
      <div class="app-muted" style="margin-top: 4px;">上传或粘贴历史数据，一键生成“下周”预测值与趋势图（线性回归）。</div>
    </div>
    <div class="app-muted" style="font-size: 12px;">Demo 级预测：用于快速估算与趋势参考</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


def _make_template_csv(granularity: Granularity) -> bytes:
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


def _parse_manual_input(raw: str) -> ParsedInput:
    values = _parse_numeric_series(raw)
    return ParsedInput(values=values, dates=None)


def _parse_csv_input(file) -> ParsedInput:
    values, dates = _read_csv_series(file)
    return ParsedInput(values=values, dates=dates)


with st.sidebar:
    st.subheader("参数")
    target_name = st.text_input("指标名称", value="客流")
    unit = st.text_input("单位（可选）", value="人")
    granularity_label = st.radio(
        "数据粒度",
        options=["按天（预测未来 7 天合计）", "按周（预测下一周）"],
        index=0,
    )
    granularity_key: Granularity = "daily" if "按天" in granularity_label else "weekly"
    st.divider()
    st.subheader("CSV 模板")
    st.download_button(
        "下载 CSV 模板",
        data=_make_template_csv(granularity_key),
        file_name=f"template_{'daily' if granularity_key == 'daily' else 'weekly'}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("CSV 两列即可：`date,value`（大小写不敏感）。")
    st.divider()
    st.subheader("示例数据")
    seed = st.number_input("示例随机种子", value=7, min_value=0, max_value=10_000)
    if st.button("生成示例并填入", use_container_width=True):
        rng = np.random.default_rng(int(seed))
        n = 365 if granularity_key == "daily" else 52
        t = np.arange(n)
        trend = 120 + 0.15 * t
        weekly = 12 * np.sin(2 * np.pi * t / 7) if granularity_key == "daily" else 0
        noise = rng.normal(0, 8 if granularity_key == "daily" else 30, size=n)
        y = np.clip(trend + weekly + noise, 0, None)
        st.session_state["manual_data"] = "\n".join(str(int(v)) for v in y)


input_tab, upload_tab = st.tabs(["数据输入", "文件上传"])

parsed: Optional[ParsedInput] = None
parse_error: Optional[str] = None

with input_tab:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("粘贴历史数据")
    raw = st.text_area(
        "数值列表（支持逗号 / 空格 / 换行分隔）",
        height=220,
        key="manual_data",
        placeholder="例如：\n120\n132\n128\n...",
    )
    col_a, col_b = st.columns([0.62, 0.38], vertical_alignment="bottom")
    with col_a:
        st.caption("建议：按天至少 30 个点、按周至少 12 个点，预测会更稳定。")
    with col_b:
        parse_manual = st.button("解析这份数据", use_container_width=True)
    if parse_manual:
        if not raw.strip():
            parse_error = "请输入或粘贴历史数据"
        else:
            try:
                parsed = _parse_manual_input(raw)
            except Exception:
                parse_error = "无法解析：请确认都是数值，并用逗号/空格/换行分隔"
    st.markdown("</div>", unsafe_allow_html=True)

with upload_tab:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("上传 CSV")
    file = st.file_uploader("选择文件", type=["csv"], label_visibility="collapsed")
    parse_upload = st.button("解析 CSV", use_container_width=True, disabled=file is None)
    if parse_upload and file is not None:
        try:
            parsed = _parse_csv_input(file)
            if parsed.dates is not None:
                inferred = _infer_granularity_from_dates(parsed.dates)
                if inferred != granularity_key:
                    st.info(
                        f"从日期间隔推断更像：{'按天' if inferred == 'daily' else '按周'}。你也可以在左侧切换粒度。"
                    )
        except Exception as e:
            parse_error = str(e)
    st.caption("CSV 两列即可（日期+数值）。推荐列名：date,value。")
    st.markdown("</div>", unsafe_allow_html=True)


if parse_error:
    st.error(parse_error)


if parsed is not None:
    values = parsed.values
    if values is not None and len(values) > 0:
        preview_n = min(12, len(values))
        st.caption(
            f"已读取 {len(values)} 个数值。预览前 {preview_n} 个：{', '.join(_format_number(v) for v in values[:preview_n])}"
        )

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("数据预览")
    if parsed.dates is not None:
        df_data = pd.DataFrame({"date": pd.to_datetime(parsed.dates), "value": values})
        st.dataframe(df_data.tail(30), use_container_width=True, hide_index=True)
    else:
        df_data = pd.DataFrame({"index": np.arange(1, len(values) + 1), "value": values})
        st.dataframe(df_data.tail(30), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("生成预测")
    run = st.button("开始预测", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        with st.spinner("正在拟合模型并生成预测..."):
            try:
                result = _fit_forecast(values, granularity_key)
            except Exception as e:
                st.error(str(e))
            else:
                unit_suffix = f" {unit}".strip() if unit.strip() else ""
                headline = (
                    f"预测：下周 {target_name}（合计）"
                    if granularity_key == "daily"
                    else f"预测：下一周 {target_name}"
                )

                st.markdown('<div class="app-card">', unsafe_allow_html=True)
                st.subheader("结果")
                st.metric(label=headline, value=f"{_format_number(result.future_total)}{unit_suffix}")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("R²（留出集）", "—" if np.isnan(result.r2) else f"{result.r2:.3f}")
                with m2:
                    st.metric("MAE（留出集）", f"{_format_number(result.mae)}{unit_suffix}")
                with m3:
                    st.metric("样本数", f"{len(result.history)}")

                if result.approx_low is not None and result.approx_high is not None:
                    st.caption(
                        f"粗略范围（基于残差波动）：{_format_number(result.approx_low)} ~ {_format_number(result.approx_high)}{unit_suffix}"
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                hist = result.history
                fitted = result.fitted
                future = result.future
                idx_hist = np.arange(len(hist))
                idx_future = np.arange(len(hist), len(hist) + len(future))
                df_plot = pd.DataFrame(
                    {
                        "历史": pd.Series(hist, index=idx_hist),
                        "拟合趋势": pd.Series(fitted, index=idx_hist),
                        "未来预测": pd.Series(future, index=idx_future),
                    }
                )

                st.markdown('<div class="app-card">', unsafe_allow_html=True)
                st.subheader("趋势图")
                st.line_chart(df_plot, height=340)
                st.markdown("</div>", unsafe_allow_html=True)

                if granularity_key == "daily":
                    detail = pd.DataFrame(
                        {
                            "day_offset": np.arange(1, 8),
                            "forecast": np.round(future, 2),
                        }
                    )
                else:
                    detail = pd.DataFrame({"week_offset": [1], "forecast": np.round(future, 2)})

                st.markdown('<div class="app-card">', unsafe_allow_html=True)
                st.subheader("明细")
                st.dataframe(detail, use_container_width=True, hide_index=True)
                st.download_button(
                    "下载预测明细 CSV",
                    data=detail.to_csv(index=False).encode("utf-8"),
                    file_name="forecast_detail.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("先在“数据输入 / 文件上传”里解析一份数据，再生成预测。")


