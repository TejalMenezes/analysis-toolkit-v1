
"""Shared look & feel for Smart Analysis Reporter — white surface, orange accents."""

import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go


# ── Brand palette ──
ORANGE = "#F57C00"
ORANGE_DARK = "#E65100"
ORANGE_SOFT = "#FFF6EE"
ORANGE_LINE = "#FFE0C2"
INK = "#1F2430"
MUTED = "#7A8290"

APP_NAME = "Smart Analysis Reporter"
AUTHOR_NAME = "Kavin Ganapathy"
AUTHOR_ID = "100008820"

# Matplotlib-friendly categorical palette (orange-led)
PALETTE = ["#F57C00", "#FF9E40", "#FFB870", "#C75B00", "#FFD2A6", "#8A5A2B"]


def _register_plotly_template():
    """An orange/white Plotly template applied app-wide."""
    tpl = go.layout.Template()
    tpl.layout.colorway = PALETTE
    tpl.layout.font = dict(color=INK, family="sans-serif")
    tpl.layout.paper_bgcolor = "#FFFFFF"
    tpl.layout.plot_bgcolor = "#FFFFFF"
    tpl.layout.xaxis = dict(gridcolor="#F0F0F0", zerolinecolor="#E8E8E8")
    tpl.layout.yaxis = dict(gridcolor="#F0F0F0", zerolinecolor="#E8E8E8")
    tpl.layout.colorscale.sequential = [
        [0.0, "#FFF6EE"], [0.5, "#FFB870"], [1.0, "#E65100"]
    ]
    pio.templates["sar"] = tpl
    pio.templates.default = "plotly+sar"


_CSS = f"""
<style>
:root {{
  --sar-orange: {ORANGE};
  --sar-orange-dark: {ORANGE_DARK};
  --sar-soft: {ORANGE_SOFT};
  --sar-line: {ORANGE_LINE};
  --sar-ink: {INK};
}}

/* page width + base — extra top padding so headings clear the toolbar */
.block-container {{ padding-top: 4rem; padding-bottom: 3rem; max-width: 1180px; }}
html, body, [class*="css"] {{ color: var(--sar-ink); }}

/* keep Streamlit's top toolbar (Deploy/menu) but make it blend in */
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ background-image: linear-gradient(90deg, {ORANGE}, #FFB870); }}

/* sidebar */
[data-testid="stSidebar"] {{ background: {ORANGE_SOFT}; border-right: 1px solid {ORANGE_LINE}; }}
[data-testid="stSidebarNav"]::before {{
  content: "Smart Analysis Reporter";
  display: block; padding: 14px 16px 4px; font-weight: 800; font-size: 15px;
  color: {ORANGE_DARK}; letter-spacing: .2px;
}}

/* brand header */
.sar-header {{
  display:flex; align-items:center; gap:16px; margin:6px 0 6px;
}}
.sar-logo {{
  width:46px; height:46px; border-radius:12px; flex-shrink:0;
  background:linear-gradient(135deg, {ORANGE}, #FF9E40);
  display:flex; align-items:center; justify-content:center;
  color:#fff; font-size:22px; font-weight:800;
  box-shadow:0 6px 16px rgba(245,124,0,.32);
}}
.sar-title {{ font-size:24px; font-weight:800; line-height:1.25; color:var(--sar-ink);
  padding-top:2px; }}
.sar-sub {{ font-size:13px; color:{MUTED}; margin-top:2px; }}
.sar-rule {{ height:3px; border:0; margin:14px 0 22px;
  background:linear-gradient(90deg, {ORANGE} 0%, #FFB870 40%, transparent 100%); border-radius:3px; }}

/* author chip */
.sar-chip {{
  display:inline-flex; align-items:center; gap:8px; padding:6px 14px; border-radius:999px;
  background:#fff; border:1px solid {ORANGE_LINE}; font-size:12.5px; color:var(--sar-ink);
  box-shadow:0 1px 3px rgba(0,0,0,.05);
}}
.sar-chip b {{ color:{ORANGE_DARK}; }}
.sar-chip .sar-id {{ color:{MUTED}; font-size:11.5px; }}

/* KPI cards */
.sar-kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:6px 0 8px; }}
.sar-kpi {{
  background:#fff; border:1px solid #EFEFEF; border-left:4px solid {ORANGE};
  border-radius:14px; padding:16px 18px; box-shadow:0 2px 8px rgba(0,0,0,.04);
}}
.sar-kpi .v {{ font-size:26px; font-weight:800; color:var(--sar-ink); line-height:1; }}
.sar-kpi .l {{ font-size:11.5px; color:{MUTED}; text-transform:uppercase; letter-spacing:.6px; margin-top:7px; }}

/* buttons → orange */
.stButton > button, .stDownloadButton > button {{
  border-radius:10px; border:1px solid {ORANGE_LINE}; font-weight:600;
}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background:{ORANGE}; border-color:{ORANGE}; color:#fff;
  box-shadow:0 4px 12px rgba(245,124,0,.28);
}}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {{
  background:{ORANGE_DARK}; border-color:{ORANGE_DARK};
}}
.stButton > button:hover {{ border-color:{ORANGE}; color:{ORANGE_DARK}; }}

/* tabs underline */
.stTabs [data-baseweb="tab-highlight"] {{ background-color:{ORANGE}; }}
.stTabs [aria-selected="true"] {{ color:{ORANGE_DARK} !important; }}

/* metric accent */
[data-testid="stMetricValue"] {{ color:var(--sar-ink); }}

/* section subheaders */
h2, h3 {{ color:var(--sar-ink); }}
</style>
"""


def setup(active: str = ""):
    """Call once at the top of every page: applies theme + Plotly template."""
    _register_plotly_template()
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = "", icon: str = "📊"):
    st.markdown(
        f"""
        <div class="sar-header">
          <div class="sar-logo">{icon}</div>
          <div>
            <div class="sar-title">{title}</div>
            <div class="sar-sub">{subtitle}</div>
          </div>
        </div>
        <hr class="sar-rule"/>
        """,
        unsafe_allow_html=True,
    )


def author_chip():
    st.markdown(
        f"""<span class="sar-chip">👤 <b>{AUTHOR_NAME}</b>
        <span class="sar-id">ID {AUTHOR_ID}</span></span>""",
        unsafe_allow_html=True,
    )


def kpi_cards(items):
    """items: list of (value, label)."""
    cards = "".join(
        f'<div class="sar-kpi"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for v, l in items
    )
    st.markdown(f'<div class="sar-kpis">{cards}</div>', unsafe_allow_html=True)
