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

    values, dates, _meta = read_csv_series(io.BytesIO(file_bytes))
    return ParsedInput(values=values, dates=dates)


@st.cache_data(show_spinner=False)
def _read_csv_df(file_bytes: bytes) -> pd.DataFrame:
    import io

    df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", sep=None, engine="python")
    df = df.dropna(axis=1, how="all")
    return df


@st.cache_data(show_spinner=False)
def _csv_candidates(file_bytes: bytes) -> tuple[list[str], list[str]]:
    df = _read_csv_df(file_bytes)
    n = len(df)
    numeric_cols: list[str] = []
    date_cols: list[str] = []
    for col in df.columns:
        s_num = pd.to_numeric(df[col], errors="coerce")
        if int(s_num.notna().sum()) >= max(5, int(0.6 * n)):
            numeric_cols.append(str(col))
        s_date = pd.to_datetime(df[col], errors="coerce")
        if int(s_date.notna().sum()) >= max(3, int(0.6 * n)):
            date_cols.append(str(col))
    return numeric_cols, date_cols


@st.cache_data(show_spinner=False)
def _infer_store_schema(file_bytes: bytes) -> dict:
    df = _read_csv_df(file_bytes)
    n = len(df)

    def _kw_any(name: str, kws: list[str]) -> bool:
        s = str(name).strip().lower()
        return any(k in s for k in kws)

    merchant_cols: list[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        valid = int(s.notna().sum())
        if valid < max(10, int(0.6 * n)):
            continue
        nunique = int(s.astype(str).nunique(dropna=True))
        if nunique <= 1:
            continue
        if _kw_any(col, ["merchant", "shop", "store", "tenant", "vendor", "商户", "门店", "店铺", "柜台"]):
            merchant_cols.append(str(col))
            continue
        if 2 <= nunique <= min(500, max(20, int(0.2 * n))):
            merchant_cols.append(str(col))

    def _ymd_pick(kws: list[str]) -> list[str]:
        cols: list[str] = []
        for col in df.columns:
            if _kw_any(col, kws):
                s_num = pd.to_numeric(df[col], errors="coerce")
                if int(s_num.notna().sum()) >= max(10, int(0.6 * n)):
                    cols.append(str(col))
        return cols

    year_cols = _ymd_pick(["year", "yyyy", "年"])
    month_cols = _ymd_pick(["month", "mm", "月"])
    day_cols = _ymd_pick(["day", "dd", "日", "date", "dayofmonth"])

    numeric_cols, datetime_cols = _csv_candidates(file_bytes)
    return {
        "merchant_cols": merchant_cols,
        "numeric_cols": numeric_cols,
        "datetime_cols": datetime_cols,
        "year_cols": year_cols,
        "month_cols": month_cols,
        "day_cols": day_cols,
        "rows": len(df),
        "cols": [str(c) for c in df.columns],
    }


def _parse_store_csv(
    file_bytes: bytes,
    *,
    value_col: str,
    merchant_col: str | None,
    date_mode: str,
    datetime_col: str | None,
    year_col: str | None,
    month_col: str | None,
    day_col: str | None,
    merchant_scope: str,
) -> tuple[ParsedInput, dict]:
    df = _read_csv_df(file_bytes)
    if value_col not in df.columns:
        raise ValueError("数值列不存在")

    values_s = pd.to_numeric(df[value_col], errors="coerce")

    dates_s: pd.Series | None = None
    if date_mode == "datetime" and datetime_col and datetime_col in df.columns:
        dates_s = pd.to_datetime(df[datetime_col], errors="coerce")
    elif date_mode == "ymd" and year_col and month_col and day_col:
        if year_col not in df.columns or month_col not in df.columns or day_col not in df.columns:
            raise ValueError("年月日列不存在")
        y = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
        m = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
        d = pd.to_numeric(df[day_col], errors="coerce").astype("Int64")
        dates_s = pd.to_datetime({"year": y, "month": m, "day": d}, errors="coerce")
    else:
        if datetime_col and datetime_col in df.columns:
            dates_s = pd.to_datetime(df[datetime_col], errors="coerce")
        elif year_col and month_col and day_col:
            y = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
            m = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
            d = pd.to_numeric(df[day_col], errors="coerce").astype("Int64")
            dates_s = pd.to_datetime({"year": y, "month": m, "day": d}, errors="coerce")

    if dates_s is None:
        raise ValueError("未检测到日期信息：请选择日期列，或选择年/月/日列")

    tmp = pd.DataFrame({"date": pd.to_datetime(dates_s, errors="coerce"), "value": values_s})
    if merchant_col and merchant_col in df.columns:
        tmp["merchant"] = df[merchant_col].astype(str)
    else:
        tmp["merchant"] = None

    tmp = tmp.dropna(subset=["date", "value"])
    if tmp.empty:
        raise ValueError("未能从所选列解析出有效日期/数值")

    tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()
    tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
    tmp = tmp.dropna(subset=["date", "value"])
    tmp["value"] = np.clip(tmp["value"].to_numpy(dtype=float), 0.0, None)

    if merchant_scope != "all" and merchant_col and merchant_col in df.columns:
        tmp = tmp[tmp["merchant"] == merchant_scope]

    daily = tmp.groupby("date", as_index=False)["value"].sum().sort_values("date")
    merchants = (
        tmp.groupby("merchant", as_index=False)["value"].sum().sort_values("value", ascending=False)
        if merchant_col and merchant_col in df.columns
        else pd.DataFrame(columns=["merchant", "value"])
    )

    parsed = ParsedInput(values=daily["value"].to_numpy(dtype=float), dates=pd.Series(daily["date"]))
    meta = {
        "rows_in_file": int(len(df)),
        "rows_used": int(len(tmp)),
        "days": int(len(daily)),
        "merchant_col": merchant_col,
        "value_col": value_col,
        "date_mode": date_mode,
        "datetime_col": datetime_col,
        "year_col": year_col,
        "month_col": month_col,
        "day_col": day_col,
        "merchant_scope": merchant_scope,
        "merchants": merchants,
    }
    return parsed, meta


def _parse_csv_with_selection(file_bytes: bytes, value_col: str, date_col: str | None) -> ParsedInput:
    df = _read_csv_df(file_bytes)
    if value_col not in df.columns:
        raise ValueError("选择的数值列不存在")

    values_s = pd.to_numeric(df[value_col], errors="coerce")
    if date_col and date_col in df.columns:
        dates_s = pd.to_datetime(df[date_col], errors="coerce")
        mask = dates_s.notna() & values_s.notna()
        dates_s = dates_s[mask]
        values_s = values_s[mask]
        if values_s.empty:
            raise ValueError("未能从所选列解析出有效数值")
        order = np.argsort(dates_s.to_numpy())
        return ParsedInput(values=values_s.to_numpy(dtype=float)[order], dates=pd.Series(dates_s.iloc[order]))

    values_s = values_s.dropna()
    if values_s.empty:
        raise ValueError("未能从所选列解析出有效数值")
    return ParsedInput(values=values_s.to_numpy(dtype=float), dates=None)


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


def _save_parsed(
    p: ParsedInput,
    source: str,
    *,
    meta: dict | None = None,
    force_granularity: str | None = None,
) -> None:
    st.session_state["parsed_source"] = source
    st.session_state["parsed_values"] = [float(x) for x in np.asarray(p.values, dtype=float).reshape(-1)]
    if p.dates is None:
        st.session_state["parsed_dates"] = None
    else:
        st.session_state["parsed_dates"] = [
            pd.to_datetime(d).strftime("%Y-%m-%d %H:%M:%S") for d in pd.to_datetime(p.dates)
        ]
    st.session_state["parsed_meta"] = meta or {}
    st.session_state["parsed_force_granularity"] = force_granularity


def _load_parsed() -> ParsedInput | None:
    if "parsed_values" not in st.session_state:
        return None
    values = np.asarray(st.session_state.get("parsed_values") or [], dtype=float)
    dates_raw = st.session_state.get("parsed_dates")
    if not values.size:
        return None
    if dates_raw:
        dates = pd.to_datetime(pd.Series(dates_raw), errors="coerce")
        return ParsedInput(values=values, dates=dates)
    return ParsedInput(values=values, dates=None)


parsed = _load_parsed()

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
                _save_parsed(parsed, source="manual", meta={"mode": "manual"}, force_granularity=None)
            except Exception:
                parse_error = "无法解析：请确认都是数值，并用逗号/空格/换行分隔"
    st.markdown("</div>", unsafe_allow_html=True)

with tab_upload:
    st.markdown('<div class="pp-card">', unsafe_allow_html=True)
    st.subheader("上传 CSV")
    file = st.file_uploader("选择文件", type=["csv"], label_visibility="collapsed")

    file_bytes: bytes | None = None
    if file is not None:
        file_bytes = file.getvalue()

    if file_bytes:
        schema = _infer_store_schema(file_bytes)
        numeric_cols = schema["numeric_cols"]
        date_cols = schema["datetime_cols"]
        merchant_cols = schema["merchant_cols"]
        y_cols = schema["year_cols"]
        m_cols = schema["month_cols"]
        d_cols = schema["day_cols"]

        store_possible = bool(date_cols) or (bool(y_cols) and bool(m_cols) and bool(d_cols))
        store_mode = st.checkbox(
            "按商户数据解析（自动按天汇总，预测下周全店客流）",
            value=bool(merchant_cols) and store_possible,
            help="适合“商户 + 人流量 + 年/月/日（或时间戳）”的明细表。系统会先把各商户按天汇总成“全店每天总客流”，再预测未来 7 天合计。",
        )

        if store_mode:
            auto_value_col = numeric_cols[0] if numeric_cols else None
            auto_datetime_col = date_cols[0] if date_cols else None
            auto_merchant_col = merchant_cols[0] if merchant_cols else None

            date_mode = "ymd" if (y_cols and m_cols and d_cols) else ("datetime" if date_cols else "auto")
            with st.expander("字段映射（可调整）", expanded=False):
                if not numeric_cols:
                    st.error("未找到可用的数值列：请确认 CSV 至少包含一列人流量/客流等数字")
                else:
                    selected_value = st.selectbox(
                        "人流量/客流列（数值列）",
                        options=numeric_cols,
                        index=numeric_cols.index(auto_value_col) if auto_value_col in numeric_cols else 0,
                    )

                    merchant_options = ["(无商户列)"] + merchant_cols
                    selected_merchant = st.selectbox(
                        "商户列（可选）",
                        options=merchant_options,
                        index=merchant_options.index(auto_merchant_col) if auto_merchant_col in merchant_options else 0,
                        help="如果存在商户列，系统会统计各商户贡献并支持按商户筛选；若不选则默认只有全店总量。",
                    )

                    mode_options = ["自动", "日期列", "年-月-日三列"]
                    selected_mode_label = st.selectbox(
                        "日期来源",
                        options=mode_options,
                        index=2 if date_mode == "ymd" else (1 if date_mode == "datetime" else 0),
                    )
                    selected_mode = {
                        "自动": "auto",
                        "日期列": "datetime",
                        "年-月-日三列": "ymd",
                    }[selected_mode_label]

                    selected_datetime = None
                    selected_year = None
                    selected_month = None
                    selected_day = None

                    if selected_mode in {"auto", "datetime"}:
                        dt_options = ["(无日期列)"] + date_cols
                        selected_datetime = st.selectbox(
                            "日期列（可选）",
                            options=dt_options,
                            index=dt_options.index(auto_datetime_col) if auto_datetime_col in dt_options else 0,
                        )
                        if selected_datetime == "(无日期列)":
                            selected_datetime = None

                    if selected_mode in {"auto", "ymd"}:
                        if y_cols and m_cols and d_cols:
                            selected_year = st.selectbox("年列", options=y_cols, index=0)
                            selected_month = st.selectbox("月列", options=m_cols, index=0)
                            selected_day = st.selectbox("日列", options=d_cols, index=0)
                        else:
                            st.caption("未检测到年/月/日三列；请改用“日期列”模式或检查表头。")

            merchant_scope = "all"
            if merchant_cols:
                try:
                    df_tmp = _read_csv_df(file_bytes)
                    if auto_merchant_col and auto_merchant_col in df_tmp.columns:
                        uniq = df_tmp[auto_merchant_col].dropna().astype(str).unique().tolist()
                        uniq = uniq[:200]
                    else:
                        uniq = []
                except Exception:
                    uniq = []
                scope_options = ["全店合计"] + uniq
                scope_sel = st.selectbox(
                    "预测范围",
                    options=scope_options,
                    index=0,
                    help="默认预测全店（所有商户）整体客流；也可以只预测某一个商户。",
                )
                merchant_scope = "all" if scope_sel == "全店合计" else scope_sel

            can_parse_store = bool(numeric_cols)
            parse_upload = st.button("解析 CSV", use_container_width=True, disabled=not can_parse_store)
            if parse_upload and can_parse_store:
                try:
                    m_col = None if "selected_merchant" not in locals() or selected_merchant == "(无商户列)" else selected_merchant
                    p, meta = _parse_store_csv(
                        file_bytes,
                        value_col=selected_value,
                        merchant_col=m_col,
                        date_mode=selected_mode,
                        datetime_col=selected_datetime if "selected_datetime" in locals() else None,
                        year_col=selected_year if "selected_year" in locals() else None,
                        month_col=selected_month if "selected_month" in locals() else None,
                        day_col=selected_day if "selected_day" in locals() else None,
                        merchant_scope=merchant_scope,
                    )

                    st.success(
                        f"已解析：数值列 = {meta['value_col']}"
                        + (f"；商户列 = {meta['merchant_col']}" if meta.get("merchant_col") else "")
                        + f"；汇总后天数 = {meta['days']}"
                    )
                    st.caption(f"原始行数：{meta['rows_in_file']}；用于汇总的有效行数：{meta['rows_used']}")

                    _save_parsed(
                        p,
                        source="csv_store",
                        meta={"mode": "csv_store", **{k: v for k, v in meta.items() if k != "merchants"}},
                        force_granularity="daily",
                    )
                    st.session_state["parsed_merchants"] = meta.get("merchants")
                    parsed = p
                except Exception as e:
                    parse_error = str(e)
        else:
            numeric_cols, date_cols = _csv_candidates(file_bytes)

            try:
                _, _, meta = read_csv_series(file)
                auto_value_col = meta.value_col
                auto_date_col = meta.date_col
            except Exception:
                auto_value_col = numeric_cols[0] if numeric_cols else None
                auto_date_col = date_cols[0] if date_cols else None

            with st.expander("手动选择列（兜底）", expanded=False):
                if not numeric_cols:
                    st.error("未找到可用的数值列：请确认 CSV 至少包含一列数字")
                else:
                    selected_value = st.selectbox(
                        "数值列",
                        options=numeric_cols,
                        index=numeric_cols.index(auto_value_col) if auto_value_col in numeric_cols else 0,
                        help="选择哪一列作为预测的历史数值。",
                    )
                    date_options = ["(无日期列)"] + date_cols
                    default_date = auto_date_col if auto_date_col in date_cols else "(无日期列)"
                    selected_date = st.selectbox(
                        "日期列（可选）",
                        options=date_options,
                        index=date_options.index(default_date) if default_date in date_options else 0,
                        help="如果选择日期列，系统会按日期排序；否则按表格原顺序使用数值。",
                    )

                    use_manual = st.checkbox(
                        "使用手动选择（覆盖自动识别）",
                        value=False,
                    )

            parse_upload = st.button("解析 CSV", use_container_width=True)
            if parse_upload:
                try:
                    if "use_manual" in locals() and use_manual:
                        date_col = None if selected_date == "(无日期列)" else selected_date
                        parsed = _parse_csv_with_selection(file_bytes, selected_value, date_col)
                        st.success(
                            f"已使用手动选择：数值列 = {selected_value}"
                            + (f"；日期列 = {date_col}" if date_col else "")
                        )
                    else:
                        values, dates, meta = read_csv_series(file)
                        parsed = ParsedInput(values=values, dates=dates)
                        if meta.date_col:
                            st.success(f"已自动识别：日期列 = {meta.date_col}；数值列 = {meta.value_col}")
                        else:
                            st.success(f"已自动识别：数值列 = {meta.value_col}（未检测到日期列）")

                    _save_parsed(parsed, source="csv", meta={"mode": "csv"}, force_granularity=None)

                    if parsed.dates is not None:
                        inferred = infer_granularity_from_dates(parsed.dates)
                        if inferred != granularity_key:
                            st.info(
                                f"从日期间隔推断更像：{'按天' if inferred == 'daily' else '按周'}。你也可以在左侧切换粒度。"
                            )
                except Exception as e:
                    parse_error = str(e)
    else:
        parse_upload = st.button("解析 CSV", use_container_width=True, disabled=True)

    st.caption("支持任意列名：系统会自动识别；支持“商户 + 年/月/日（或日期列）+ 人流量”并自动按天汇总预测下周。")
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

merchants_df = st.session_state.get("parsed_merchants")
if isinstance(merchants_df, pd.DataFrame) and not merchants_df.empty:
    show_n = min(10, len(merchants_df))
    st.caption(f"商户贡献（Top {show_n}，按汇总客流排序）：")
    preview = merchants_df.head(show_n).copy()
    preview.columns = ["商户", "汇总客流"]
    st.dataframe(preview, use_container_width=True, hide_index=True)
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
        non_decreasing = bool((diffs >= pd.Timedelta(0)).all())
        if not non_decreasing:
            dates_ok = False
            st.error("日期顺序不能倒序（旧 → 新）。允许重复日期，但不要把顺序改乱。")
        else:
            if bool(edited_dates.duplicated().any()):
                st.warning("检测到重复日期：系统允许重复日期，预测会按你的原始行顺序使用这些记录。")

            st.session_state[override_key] = list(edited_dates.dt.date)
            parsed = ParsedInput(values=values, dates=pd.Series(pd.to_datetime(edited_dates)))
            _save_parsed(parsed, source=str(st.session_state.get("parsed_source") or "edited"))
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
granularity_effective = st.session_state.get("parsed_force_granularity") or granularity_key
if granularity_effective != granularity_key:
    st.info("当前数据已按天汇总，将按“预测未来 7 天合计”的口径输出下周预测。")

with st.spinner("正在拟合模型并生成预测..."):
    result = fit_forecast(
        values=values,
        granularity=granularity_effective,
        holdout_points=holdout_points,
        bootstrap_trials=bootstrap_trials,
        bootstrap_alpha=bootstrap_alpha,
    )

headline = (
    f"预测：下周 {target_name}（合计）"
    if granularity_effective == "daily"
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
    hist_idx_s = pd.Series(hist_idx)
    dup_offset = hist_idx_s.groupby(hist_idx_s).cumcount()
    hist_idx_plot = pd.DatetimeIndex(hist_idx_s + pd.to_timedelta(dup_offset, unit="ms"))
    step_days = 1 if granularity_key == "daily" else 7
    freq = "D" if granularity_key == "daily" else "7D"
    future_start = pd.to_datetime(hist_idx.iloc[-1]) + timedelta(days=step_days)
    future_idx = pd.date_range(start=future_start, periods=len(future), freq=freq)

    plot_index = pd.DatetimeIndex(list(hist_idx_plot) + list(future_idx))
    df_plot = pd.DataFrame(index=plot_index)
    df_plot.loc[hist_idx_plot, "历史"] = hist
    df_plot.loc[hist_idx_plot, "拟合趋势"] = fitted
    df_plot.loc[future_idx, "未来预测"] = future
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

    export_hist = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hist_idx),
            "date": pd.to_datetime(hist_idx).dt.strftime("%Y-%m-%d"),
            "history": np.asarray(hist, dtype=float),
            "fitted": np.asarray(fitted, dtype=float),
            "forecast": np.nan,
        }
    )
    export_future = pd.DataFrame(
        {
            "datetime": future_idx,
            "date": future_idx.strftime("%Y-%m-%d"),
            "history": np.nan,
            "fitted": np.nan,
            "forecast": np.asarray(future, dtype=float),
        }
    )
    export_series = pd.concat([export_hist, export_future], ignore_index=True)
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

