from __future__ import annotations

import streamlit as st

from ui import render_top_nav


st.set_page_config(
    page_title="PredictPro 预测工具",
    page_icon="📈",
    layout="wide",
)

render_top_nav(active="home")

st.markdown(
    """
<style>
  .block-container { padding-top: 1.15rem; padding-bottom: 2.1rem; }
  .pp-hero { border: 1px solid rgba(15, 23, 42, 0.08); background: radial-gradient(1200px 600px at 10% 0%, rgba(46,107,230,0.22), rgba(255,255,255,0)), linear-gradient(180deg, rgba(46,107,230,0.10), rgba(255,255,255,0)); padding: 22px 22px; border-radius: 18px; }
  .pp-card { border: 1px solid rgba(15, 23, 42, 0.08); background: #ffffff; padding: 16px 18px; border-radius: 16px; height: 100%; }
  .pp-muted { color: rgba(15, 23, 42, 0.70); }
  .pp-kpi { font-size: 36px; font-weight: 800; letter-spacing: -0.02em; }
  .pp-badge { display: inline-block; padding: 4px 10px; border-radius: 999px; background: rgba(46,107,230,0.12); color: rgba(46,107,230,1); font-weight: 600; font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.title("PredictPro")
st.sidebar.caption("线性回归 · 快速预测")

st.markdown(
    """
<div class="pp-hero">
  <div class="pp-badge">Quick Forecast</div>
  <div style="display:flex; justify-content:space-between; align-items:flex-end; gap: 18px; flex-wrap: wrap; margin-top: 10px;">
    <div>
      <div style="font-size: 28px; font-weight: 800;">让下周预测变得简单、专业、可解释</div>
      <div class="pp-muted" style="margin-top: 6px; font-size: 14px; max-width: 880px;">
        把去年的客流/销售数据粘贴或上传为 CSV，系统会用线性回归拟合趋势并给出下周预测，同时展示基线对比、误差指标与区间范围。
      </div>
    </div>
    <div style="min-width: 320px;">
      <div class="pp-muted" style="font-size: 12px;">建议数据量</div>
      <div class="pp-kpi">≥ 30 天 / 12 周</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        """
<div class="pp-card">
  <div style="font-size: 16px; font-weight: 700;">清晰输入</div>
  <div class="pp-muted" style="margin-top: 6px;">支持粘贴数值或上传 CSV（date,value）。自动预览、校验与快速纠错。</div>
</div>
""",
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        """
<div class="pp-card">
  <div style="font-size: 16px; font-weight: 700;">专业输出</div>
  <div class="pp-muted" style="margin-top: 6px;">展示预测值、趋势图、R²/MAE、与“历史基线”对比，支持下载预测明细。</div>
</div>
""",
        unsafe_allow_html=True,
    )
with col_c:
    st.markdown(
        """
<div class="pp-card">
  <div style="font-size: 16px; font-weight: 700;">可解释方法</div>
  <div class="pp-muted" style="margin-top: 6px;">线性回归拟合趋势，提供趋势斜率与基于残差的区间估计（用于展示波动范围）。</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")

cta_left, cta_right = st.columns([0.72, 0.28], vertical_alignment="center")
with cta_left:
    st.subheader("立即开始")
    st.caption("进入预测器页面，上传或粘贴数据即可生成下周预测。")
with cta_right:
    st.page_link("pages/1_预测器.py", label="进入预测器", icon="📈", use_container_width=True)

st.divider()

st.caption(
    "免责声明：本工具用于快速估算与趋势参考，不构成经营或财务建议。"
)

