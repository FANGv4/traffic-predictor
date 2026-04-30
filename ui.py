from __future__ import annotations

import streamlit as st


def render_top_nav(active: str) -> None:
    st.markdown(
        """
<style>
  .pp-topnav { border: 1px solid rgba(15, 23, 42, 0.08); background: rgba(255,255,255,0.88); backdrop-filter: blur(10px); border-radius: 16px; padding: 10px 12px; }
  .pp-topnav-title { font-weight: 800; letter-spacing: -0.01em; }
  .pp-topnav-sub { color: rgba(15, 23, 42, 0.62); font-size: 12px; margin-top: 2px; }
  .pp-topnav-active { display: inline-block; padding: 3px 10px; border-radius: 999px; background: rgba(46,107,230,0.12); color: rgba(46,107,230,1); font-weight: 700; font-size: 12px; }
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="pp-topnav">', unsafe_allow_html=True)
    left, mid, right = st.columns([0.48, 1.25, 0.27], vertical_alignment="center")
    with left:
        st.markdown('<div class="pp-topnav-title">PredictPro</div>', unsafe_allow_html=True)
        st.markdown('<div class="pp-topnav-sub">客流/销售快速预测</div>', unsafe_allow_html=True)
    with mid:
        c1, c2, c3 = st.columns([0.9, 1.0, 1.5], gap="small")
        with c1:
            st.page_link("app.py", label="首页", icon="🏠", use_container_width=True, width="stretch")
        with c2:
            st.page_link("pages/1_预测器.py", label="预测器", icon="📈", use_container_width=True, width="stretch")
        with c3:
            st.page_link("pages/2_方法与指标.py", label="方法与指标", icon="🧪", use_container_width=True, width="stretch")
    with right:
        label = {
            "home": "首页",
            "predictor": "预测器",
            "method": "方法与指标",
        }.get(active, "")
        if label:
            st.markdown(f'<div style="text-align:right"><span class="pp-topnav-active">{label}</span></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
