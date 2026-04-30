from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from core import (
    ParsedInput,
    fit_forecast,
    format_number,
    infer_granularity_from_dates,
    make_template_csv,
    parse_numeric_series,
    read_csv_series,
)

from ui import render_top_nav


st.set_page_config(page_title="预测器 · PredictPro", page_icon="📈", layout="wide")

render_top_nav(active="predictor")

st.markdown(
    """
<style>
  .block-container { padding-top: 1.15rem; padding-bottom: 2rem; }
  .pp-card { border: 1px solid rgba(15, 23, 42, 0.08); background: #ffffff; padding: 16px 18px; border-radius: 16px; }
  .pp-muted { color: rgba(15, 23, 42, 0.70); }
  [data-testid="stMetricValue"] { font-size: 1.65rem; }
  .pp-alert-danger { border: 1px solid rgba(239, 68, 68, 0.35); background: rgba(239, 68, 68, 0.10); border-radius: 14px; padding: 10px 12px; }
  .pp-alert-danger-title { color: rgb(185, 28, 28); font-weight: 800; font-size: 13px; }
  .pp-alert-danger-body { color: rgba(15, 23, 42, 0.82); font-size: 13px; margin-top: 4px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _parse_manual(raw: str, start_date: date, granularity: str) -> ParsedInput:
    values = parse_numeric_series(raw)
    freq = "D" if granularity == "daily" else "7D"
    dates = pd.date_range(start=pd.to_datetime(start_date), periods=len(values), freq=freq)
    return ParsedInput(values=values, dates=pd.Series(dates))


@st.cache_data(show_spinner=False)
def _parse_csv(file_bytes: bytes) -> ParsedInput:
    import io

    values, dates = read_csv_series(io.BytesIO(file_bytes))
    return ParsedInput(values=values, dates=dates)


def _unit_suffix(unit: str) -> str:
    return f" {unit}" if unit.strip() else ""


st.sidebar.title("PredictPro")
st.sidebar.caption("输入数据 → 预览 → 预测 → 下载")

st.title("预测器")
st.caption("粘贴数值或上传 CSV，一键生成下周预测与趋势图。")

with st.sidebar:
    st.subheader("业务参数")
    target_name = st.text_input("指标名称", value="客流")
    unit = st.text_input("单位（可选）", value="人")
    granularity_label = st.radio(
        "数据粒度",
        options=["按天（预测未来 7 天合计）", "按周（预测下一周）"],
        index=0,
    )
    granularity_key = "daily" if "按天" in granularity_label else "weekly"

    st.subheader("评估与区间")
    holdout_points = st.slider("留出评估点数", min_value=7, max_value=60, value=14)
    bootstrap_trials = st.slider("区间试验次数", min_value=0, max_value=2000, value=800, step=100)
    interval_level = st.select_slider(
        "区间水平",
        options=[0.80, 0.90, 0.95],
        value=0.90,
        format_func=lambda x: f"{int(x * 100)}%",
    )
    bootstrap_alpha = float(1 - interval_level)

    st.divider()
    st.subheader("CSV 模板")
    st.download_button(
        "下载 CSV 模板",
        data=make_template_csv(granularity_key),
        file_name=f"template_{granularity_key}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("两列：date,value（大小写不敏感）。")


tab_input, tab_upload = st.tabs(["数据输入", "文件上传"])

parsed: ParsedInput | None = None
parse_error: str | None = None

with tab_input:
    st.markdown('<div class="pp-card">', unsafe_allow_html=True)
    st.subheader("粘贴历史数据")
    st.caption("按天：每个数字=一天；按周：每个数字=一周。按时间顺序从旧到新粘贴即可。")

    with st.expander("查看示例与常见格式", expanded=True):
        st.markdown(
            """
**分隔符是什么？有什么用？**

你粘贴的是一串数字，系统需要知道“每个数字在哪里结束、下一个数字从哪里开始”。
所以需要用 **分隔符** 把数字隔开。

在本工具里：

- **换行分隔**：每一行一个数字（最常见，适合从 Excel/表格“复制一列”直接粘贴）
- **逗号分隔**：用英文逗号 `,` 把数字隔开（适合一整行数据）
- 也支持用空格隔开（原理同上）

两种写法对预测结果 **没有区别**，只是输入习惯不同。

**数字代表什么？**

- 你在左侧选择“按天”时：每个数字表示 **一天的客流/销量**（例如 365 个数字≈一年的每天）
- 你选择“按周”时：每个数字表示 **一周的总客流/总销量**（例如 52 个数字≈一年的每周）
- 请按时间顺序粘贴：从更早的数据到最近的数据（旧 → 新）

**你可以粘贴以下任意一种：**

- 换行分隔：
"""
        )
        st.code("120\n132\n128\n140\n...", language="text")
        st.markdown("- 逗号分隔：")
        st.code("120, 132, 128, 140, ...", language="text")
        st.markdown(
            """
**常见错误：**

- 包含日期（如 `2025-01-01,120`）→ 请改用“文件上传”页签上传 CSV
- 包含单位（如 `120人`）→ 只保留数字
"""
        )

    start_date_label = "第一天日期（最早的一天）" if granularity_key == "daily" else "第一周起始日（最早的一周）"
    start_date_help = (
        "例如你粘贴了 365 个按天数据，这里填第 1 个数字对应的日期，后面的日期会自动 +1 天生成。"
        if granularity_key == "daily"
        else "例如你粘贴了 52 个按周数据，这里填第 1 个数字对应的周起始日，后面的日期会自动 +7 天生成。"
    )
    start_date_value = st.date_input(
        start_date_label,
        value=st.session_state.get("manual_start_date", date.today() - timedelta(days=30)),
        help=start_date_help,
    )
    st.session_state["manual_start_date"] = start_date_value

    left_in, right_preview = st.columns([0.62, 0.38], gap="large")

    with left_in:
        raw = st.text_area(
            "把数字粘贴到这里",
            height=240,
            key="manual_data",
            placeholder="例如：\n120\n132\n128\n140\n...",
            help="支持逗号/空格/换行分隔。只解析纯数字。",
            label_visibility="visible",
        )

    live_hint: str | None = None
    preview_df: pd.DataFrame | None = None
    if raw.strip():
        try:
            preview_vals = parse_numeric_series(raw).astype(float)
            live_hint = f"检测到 {len(preview_vals)} 个数值，预览：{', '.join(format_number(float(v)) for v in preview_vals[:8])}"
            date_col = "日期" if granularity_key == "daily" else "周起始日"
            freq = "D" if granularity_key == "daily" else "7D"
            dates = pd.date_range(start=pd.to_datetime(start_date_value), periods=len(preview_vals), freq=freq)
            preview_df = pd.DataFrame(
                {
                    date_col: dates.strftime("%Y-%m-%d"),
                    target_name: preview_vals,
                }
            )
        except Exception:
            live_hint = "当前内容暂无法解析：请确认只包含数字，并用逗号/空格/换行分隔"

    with right_preview:
        st.markdown("**日期预览**")
        st.caption("根据你填写的第一天日期自动生成，帮助你检查顺序是否正确（旧 → 新）。")
        if preview_df is None:
            st.info("粘贴数字后，这里会显示日期预览。")
        else:
            preview_limit = min(60, len(preview_df))
            st.dataframe(
                preview_df.head(preview_limit),
                use_container_width=True,
                hide_index=True,
                height=260,
            )
            if len(preview_df) > preview_limit:
                st.caption(f"仅展示前 {preview_limit} 条（共 {len(preview_df)} 条）。")

    col_l, col_r = st.columns([0.65, 0.35], vertical_alignment="bottom")
    with col_l:
        st.caption("建议：按天 ≥ 30 点；按周 ≥ 12 点。")
        if live_hint:
            st.caption(live_hint)
    with col_r:
        parse_btn = st.button("继续 → 预览数据", use_container_width=True)

    if parse_btn:
        if not raw.strip():
            parse_error = "请输入或粘贴历史数据"
        else:
            try:
                parsed = _parse_manual(raw, start_date_value, granularity_key)
            except Exception:
                parse_error = "无法解析：请确认都是数值，并用逗号/空格/换行分隔"
    st.markdown("</div>", unsafe_allow_html=True)

with tab_upload:
    st.markdown('<div class="pp-card">', unsafe_allow_html=True)
    st.subheader("上传 CSV")
    file = st.file_uploader("选择文件", type=["csv"], label_visibility="collapsed")
    parse_upload = st.button("解析 CSV", use_container_width=True, disabled=file is None)
    if parse_upload and file is not None:
        try:
            parsed = _parse_csv(file.getvalue())
            if parsed.dates is not None:
                inferred = infer_granularity_from_dates(parsed.dates)
                if inferred != granularity_key:
                    st.info(
                        f"从日期间隔推断更像：{'按天' if inferred == 'daily' else '按周'}。你也可以在左侧切换粒度。"
                    )
        except Exception as e:
            parse_error = str(e)
    st.caption("推荐列名：date,value。也支持前两列是日期与数值。")
    st.markdown("</div>", unsafe_allow_html=True)

if parse_error:
    st.error(parse_error)


if parsed is None:
    st.info("先在上方解析一份数据，再生成预测。")
    st.stop()

values = parsed.values
values = np.asarray(values, dtype=float)

st.markdown('<div class="pp-card">', unsafe_allow_html=True)
st.subheader("数据质量")

negatives = int(np.sum(values < 0))
nan_count = int(np.sum(~np.isfinite(values)))

q1, q2, q3, q4 = st.columns(4)
with q1:
    st.metric("样本数", f"{len(values)}")
with q2:
    st.metric("最小值", format_number(float(np.nanmin(values))))
with q3:
    st.metric("中位数", format_number(float(np.nanmedian(values))))
with q4:
    st.metric("最大值", format_number(float(np.nanmax(values))))

warns = []
if nan_count > 0:
    warns.append(f"包含 {nan_count} 个 NaN/inf，请清洗后再试")
if negatives > 0:
    warns.append(f"包含 {negatives} 个负数，模型会自动截断到 0")
if len(values) < (30 if granularity_key == "daily" else 12):
    warns.append("数据点偏少，预测不稳定的概率会更高")
if warns:
    st.warning("；".join(warns))
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="pp-card">', unsafe_allow_html=True)
st.subheader("数据预览")
dates_ok = True

if parsed.dates is not None:
    freq = "D" if granularity_key == "daily" else "7D"
    date_label = "日期" if granularity_key == "daily" else "周起始日"
    override_key = f"override_dates_{granularity_key}_{len(values)}"

    base_dates = pd.to_datetime(parsed.dates, errors="coerce")
    base_dates = base_dates.dt.date if hasattr(base_dates, "dt") else pd.to_datetime(base_dates).date
    dates_to_use = (
        st.session_state.get(override_key)
        if isinstance(st.session_state.get(override_key), list)
        and len(st.session_state.get(override_key)) == len(values)
        else list(base_dates)
    )

    st.markdown(
        """
<div class="pp-alert-danger">
  <div class="pp-alert-danger-title">重要提示</div>
  <div class="pp-alert-danger-body">你可以在下表直接修改日期。请确保日期按时间顺序排列（旧 → 新），不要改乱顺序。</div>
</div>
""",
        unsafe_allow_html=True,
    )

    control_left, control_right = st.columns([0.62, 0.38], vertical_alignment="bottom")
    with control_left:
        first_date = st.date_input(
            f"快速调整第一条{date_label}",
            value=dates_to_use[0] if dates_to_use else date.today(),
            help="只想改整体起点时，用这个更快；点击右侧按钮会自动按天/按周生成后续日期。",
        )
    with control_right:
        if st.button("按第一条日期重算全部", use_container_width=True):
            regenerated = pd.date_range(start=pd.to_datetime(first_date), periods=len(values), freq=freq)
            st.session_state[override_key] = list(regenerated.date)
            dates_to_use = list(regenerated.date)

    df_data = pd.DataFrame(
        {
            date_label: dates_to_use,
            "数值": values,
        }
    )

    editor_key = f"date_editor_{granularity_key}_{len(values)}"
    edited = st.data_editor(
        df_data,
        key=editor_key,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            date_label: st.column_config.DateColumn(date_label, format="YYYY-MM-DD"),
            "数值": st.column_config.NumberColumn("数值", format="%.2f"),
        },
        disabled=["数值"],
        height=420,
    )

    edited_dates = pd.to_datetime(edited[date_label], errors="coerce")
    if edited_dates.isna().any():
        dates_ok = False
        st.error("日期里存在空值/无效值，请修正后再继续。")
    else:
        diffs = edited_dates.diff().dropna()
        strictly_increasing = bool((diffs > pd.Timedelta(0)).all())
        if not strictly_increasing:
            dates_ok = False
            st.error("日期顺序必须严格递增（旧 → 新）。请不要打乱顺序，也不要出现重复日期。")
        else:
            st.session_state[override_key] = list(edited_dates.dt.date)
            parsed = ParsedInput(values=values, dates=pd.Series(pd.to_datetime(edited_dates)))
else:
    df_data = pd.DataFrame({"序号": np.arange(1, len(values) + 1), "数值": values})
    st.dataframe(df_data.tail(60), use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="pp-card">', unsafe_allow_html=True)
st.subheader("生成预测")
run = st.button("开始预测", type="primary", use_container_width=True, disabled=not dates_ok)
st.caption("提示：区间是基于留出集残差的自助采样估计，用于展示波动范围。")
st.markdown("</div>", unsafe_allow_html=True)

if not run:
    st.stop()

unit_suffix = _unit_suffix(unit)

with st.spinner("正在拟合模型并生成预测..."):
    result = fit_forecast(
        values=values,
        granularity=granularity_key,
        holdout_points=holdout_points,
        bootstrap_trials=bootstrap_trials,
        bootstrap_alpha=bootstrap_alpha,
    )

headline = (
    f"预测：下周 {target_name}（合计）"
    if granularity_key == "daily"
    else f"预测：下一周 {target_name}"
)

st.markdown('<div class="pp-card">', unsafe_allow_html=True)
st.subheader("结果")
st.metric(label=headline, value=f"{format_number(result.future_total)}{unit_suffix}")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("基线（最近）", f"{format_number(result.baseline_future_total)}{unit_suffix}")
with m2:
    delta = result.future_total - result.baseline_future_total
    st.metric("相对基线变化", f"{format_number(delta)}{unit_suffix}")
with m3:
    st.metric("R²（留出集）", "—" if np.isnan(result.r2) else f"{result.r2:.3f}")
with m4:
    st.metric("MAE（留出集）", f"{format_number(result.mae)}{unit_suffix}")

st.caption(f"趋势斜率（每步增量）：{result.trend_slope:+.4f}")

if result.boot_low is not None and result.boot_high is not None:
    st.caption(
        f"区间（{int(interval_level * 100)}%）：{format_number(result.boot_low)} ~ {format_number(result.boot_high)}{unit_suffix}"
    )

st.markdown("</div>", unsafe_allow_html=True)


hist = result.history
fitted = result.fitted
future = result.future

if parsed.dates is not None:
    hist_idx = pd.to_datetime(parsed.dates)
    step_days = 1 if granularity_key == "daily" else 7
    freq = "D" if granularity_key == "daily" else "7D"
    future_start = pd.to_datetime(hist_idx.iloc[-1]) + timedelta(days=step_days)
    future_idx = pd.date_range(start=future_start, periods=len(future), freq=freq)

    hist_s = pd.Series(hist, index=hist_idx)
    fitted_s = pd.Series(fitted, index=hist_idx)
    future_s = pd.Series(future, index=future_idx)

    df_plot = pd.DataFrame({"历史": hist_s, "拟合趋势": fitted_s, "未来预测": future_s})
else:
    idx_hist = np.arange(len(hist))
    idx_future = np.arange(len(hist), len(hist) + len(future))
    df_plot = pd.DataFrame(
        {
            "历史": pd.Series(hist, index=idx_hist),
            "拟合趋势": pd.Series(fitted, index=idx_hist),
            "未来预测": pd.Series(future, index=idx_future),
        }
    )

st.markdown('<div class="pp-card">', unsafe_allow_html=True)
st.subheader("趋势图")
st.line_chart(df_plot, height=360)
st.markdown("</div>", unsafe_allow_html=True)


if parsed.dates is not None:
    if granularity_key == "daily":
        detail = pd.DataFrame(
            {
                "date": future_idx.strftime("%Y-%m-%d"),
                "day_offset": np.arange(1, 8),
                "forecast": np.round(future, 2),
            }
        )
    else:
        detail = pd.DataFrame(
            {
                "week_start": future_idx.strftime("%Y-%m-%d"),
                "week_offset": [1],
                "forecast": np.round(future, 2),
            }
        )

    all_dates = pd.DatetimeIndex(list(hist_idx) + list(future_idx))
    export_series = pd.DataFrame(
        {
            "date": all_dates.strftime("%Y-%m-%d"),
            "history": pd.Series(hist_s).reindex(all_dates).to_numpy(),
            "fitted": pd.Series(fitted_s).reindex(all_dates).to_numpy(),
            "forecast": pd.Series(future_s).reindex(all_dates).to_numpy(),
        }
    )
else:
    if granularity_key == "daily":
        detail = pd.DataFrame(
            {
                "day_offset": np.arange(1, 8),
                "forecast": np.round(future, 2),
            }
        )
    else:
        detail = pd.DataFrame({"week_offset": [1], "forecast": np.round(future, 2)})

    idx_hist = np.arange(len(hist))
    idx_future = np.arange(len(hist), len(hist) + len(future))
    all_idx = np.concatenate([idx_hist, idx_future])
    export_series = pd.DataFrame(
        {
            "t": all_idx,
            "history": pd.Series(hist, index=idx_hist).reindex(all_idx).to_numpy(),
            "fitted": pd.Series(fitted, index=idx_hist).reindex(all_idx).to_numpy(),
            "forecast": pd.Series(future, index=idx_future).reindex(all_idx).to_numpy(),
        }
    )

st.markdown('<div class="pp-card">', unsafe_allow_html=True)
st.subheader("导出")
col_d1, col_d2 = st.columns(2)
with col_d1:
    st.download_button(
        "下载预测明细 CSV",
        data=detail.to_csv(index=False).encode("utf-8"),
        file_name="forecast_detail.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col_d2:
    st.download_button(
        "下载趋势序列 CSV",
        data=export_series.to_csv(index=False).encode("utf-8"),
        file_name="forecast_series.csv",
        mime="text/csv",
        use_container_width=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

