from __future__ import annotations

import streamlit as st

from ui import render_top_nav


st.set_page_config(page_title="方法与指标 · PredictPro", page_icon="🧪", layout="wide")

render_top_nav(active="method")

st.sidebar.title("PredictPro")
st.sidebar.caption("方法说明")

st.title("方法与指标")
st.caption("帮助你把预测结果讲清楚：模型做了什么、指标怎么读、适用边界在哪里。")


st.subheader("模型方法")
st.markdown(
    """
- 本工具使用 **线性回归** 拟合历史序列的长期趋势：`y = a·t + b`
- `t` 为时间序号（从 0 开始递增），`y` 为客流/销售等指标
- **按天**：预测未来 7 天并求和得到“下周合计”
- **按周**：预测下一周得到“下一周值”
"""
)


st.subheader("基线对比")
st.markdown(
    """
为了让预测更“可解释”，工具会同时给出一个简单基线：

- **按天**：基线 = 最近 7 天的合计
- **按周**：基线 = 最近 1 周的值

“相对基线变化”= 预测值 − 基线值，用于快速判断趋势是增长还是回落。
"""
)


st.subheader("指标含义")
st.markdown(
    """
- **R²（留出集）**：越接近 1 越好，表示趋势模型对留出数据的解释度。数据波动大时 R² 可能很低甚至为负。
- **MAE（留出集）**：平均绝对误差，单位与原始数据一致，越小越好。
- **趋势斜率**：每一步（每天/每周）平均变化量，正数代表上升趋势，负数代表下降趋势。
"""
)


st.subheader("区间范围（波动展示）")
st.markdown(
    """
区间不是严格统计置信区间，它基于“留出集残差”的自助采样：

- 先在留出集上得到残差（真实值 − 预测值）
- 抽样这些残差，加到未来预测上，得到很多条可能的未来结果
- 根据分位数得到区间（例如 90% 区间）

用途：用一个直观范围表达“未来波动可能在什么区间内”。
"""
)


st.subheader("适用边界")
st.markdown(
    """
- 适合：想要一个“趋势外推 + 快速估算”的简洁工具
- 不适合：强季节性（例如周末/节假日）、促销活动、突发事件等带来的结构性变化
""" 
)

st.caption("如果你需要更专业的时间序列模型（季节性/节假日/多变量回归），可以在此基础上升级。")

