from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Evaluation Results", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(to right, #BEB6AA, #C8C0B4, #A89E92) !important;
    border-right: 18px solid #1A1020 !important;
    box-shadow: inset -20px 0 40px rgba(0,0,0,0.28) !important;
}
[data-testid="stSidebar"] * { color: #2E2820 !important; font-size: 15px !important; }
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(to right, #2A2040 0%, #17112E 45%, #09080F 100%) !important;
    box-shadow: inset -24px 0 60px rgba(0,0,0,0.55) !important;
}
[data-testid="stAppViewContainer"] > .main .block-container {
    padding-top: 2rem !important;
}
html, body, [data-testid="stAppViewContainer"] { color: rgba(200,195,225,0.85) !important; }
h1, h2, h3 { color: rgba(220,215,240,0.92) !important; font-weight: 500 !important; }
p, span, label { font-size: 15px !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
[data-testid="stSelectbox"] > div {
    background: rgba(20,14,36,0.5) !important;
    border: 0.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: rgba(200,195,225,0.85) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:36px; font-weight:600; color:#F0C060; margin-bottom:4px;'>📊 RAGAS Evaluation Results</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:15px; color:rgba(200,195,225,0.6); margin-bottom:24px;'>"
    "11 MLflow experiments · 23 questions · 4 RAGAS metrics · hover to explore</div>",
    unsafe_allow_html=True
)

METRICS = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"]
LINE_COLORS = ["#7C3AED", "#E8844A", "#2ECC71", "#38BDF8"]
BAR_COLORS = ["#7C3AED", "#E8844A", "#2ECC71", "#38BDF8", "#F472B6"]

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(200,195,225,0.85)", size=12),
    legend=dict(
        bgcolor="rgba(30,20,50,0.8)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        font=dict(size=12),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        tickfont=dict(color="rgba(200,195,225,0.75)"),
        showline=True,
        linecolor="rgba(255,255,255,0.1)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        tickfont=dict(color="rgba(200,195,225,0.75)"),
        showline=True,
        linecolor="rgba(255,255,255,0.1)",
        range=[0.0, 1.08],
    ),
    margin=dict(l=50, r=20, t=30, b=50),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(20,14,36,0.95)",
        bordercolor="rgba(124,58,237,0.6)",
        font=dict(color="rgba(200,195,225,0.9)", size=13),
    ),
)

experiment_data = {
    "Run": [
        "01 Baseline (500/50)", "02 chunk=200/20", "03 chunk=750/75",
        "04 chunk=1000/50", "05 overlap=150", "06 Cohere reranker",
        "07 Claude Haiku", "08 rerank_n=3", "09 rerank_n=7",
        "10 LangGraph (500)", "11 LangGraph (750)",
    ],
    "Faithfulness":      [0.9416, 0.9187, 0.9779, 0.9565, 0.8790, 0.9366, 0.9290, 0.9630, 0.9384, 0.9503, 0.9299],
    "Answer Relevancy":  [0.9300, 0.8045, 0.9748, 0.9699, 0.8946, 0.9310, 0.7214, 0.9311, 0.9336, 0.9336, 0.9344],
    "Context Recall":    [0.9275, 0.7391, 0.9058, 0.9275, 0.8188, 0.9275, 0.9275, 0.9203, 0.8841, 0.9130, 0.9058],
    "Context Precision": [0.7838, 0.5200, 0.8069, 0.8554, 0.7841, 0.8090, 0.7763, 0.7742, 0.7615, 0.7885, 0.8280],
}
df_exp = pd.DataFrame(experiment_data).set_index("Run")

# --- Summary metric cards ---
cols = st.columns(4)
card_colors = ["#7C3AED", "#E8844A", "#2ECC71", "#38BDF8"]
for col, metric, color in zip(cols, METRICS, card_colors):
    best_val = df_exp[metric].max()
    best_run = df_exp[metric].idxmax()
    short_run = best_run.split(" ", 1)[1] if " " in best_run else best_run
    with col:
        st.markdown(f"""
        <div style='background: rgba(42,32,64,0.5);
                    border: 0.5px solid {color}44;
                    border-top: 3px solid {color};
                    border-radius: 10px;
                    padding: 16px;
                    margin-bottom: 24px;'>
            <div style='font-size:12px; color:rgba(200,195,225,0.55); margin-bottom:4px;'>{metric}</div>
            <div style='font-size:28px; font-weight:600; color:{color};'>{best_val:.4f}</div>
            <div style='font-size:11px; color:rgba(200,195,225,0.45); margin-top:4px;'>Best: {short_run}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin-bottom:24px;'>", unsafe_allow_html=True)

# --- Section helper ---
def section_header(title, narrative):
    st.markdown(
        f"<div style='font-size:20px; font-weight:600; color:rgba(220,215,240,0.92); margin-bottom:4px;'>{title}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='font-size:14px; color:rgba(200,195,225,0.55); margin-bottom:12px;'>{narrative}</div>",
        unsafe_allow_html=True
    )

def note_box(note, color="#7C3AED"):
    st.markdown(
        f"<div style='background:rgba(124,58,237,0.08); border-left:3px solid {color}88; "
        f"border-radius:4px; padding:10px 14px; font-size:13px; color:rgba(200,195,225,0.7); "
        f"margin:8px 0 24px;'>💡 {note}</div>",
        unsafe_allow_html=True
    )

def plotly_bar(data, y_min=0.0):
    labels = list(data.keys())
    fig = go.Figure()
    for i, (label, vals) in enumerate(zip(labels, data.values())):
        fig.add_trace(go.Bar(
            name=label,
            x=METRICS,
            y=vals,
            marker_color=BAR_COLORS[i % len(BAR_COLORS)],
            marker_opacity=0.85,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(size=11, color="rgba(200,195,225,0.8)"),
            hovertemplate=f"<b>{label}</b><br>%{{x}}: <b>%{{y:.4f}}</b><extra></extra>",
        ))
    layout = {**PLOT_LAYOUT, "barmode": "group"}
    layout["yaxis"] = dict(
        gridcolor="rgba(255,255,255,0.05)",
        tickfont=dict(color="rgba(200,195,225,0.75)"),
        range=[y_min, 1.1],
        showline=True,
        linecolor="rgba(255,255,255,0.1)",
    )
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig

# --- 1. Model Comparison ---
section_header(
    "1 · Model Comparison",
    "GPT-4o-mini (two prompt variants) vs Claude Haiku across all four RAGAS metrics."
)
st.plotly_chart(plotly_bar({
    "GPT-4o-mini":    [0.9016, 0.9356, 0.9058, 0.7996],
    "GPT-4o-mini v2": [0.9040, 0.7618, 0.9130, 0.7776],
    "Claude (Haiku)": [0.9114, 0.7282, 0.9130, 0.8093],
}), use_container_width=True)
note_box("GPT-4o-mini leads on Answer Relevancy. Claude edges ahead on Faithfulness and Context Precision.")

# --- 2. Prompt Variants ---
section_header(
    "2 · Prompt Variants (GPT-4o-mini)",
    "Three prompt strategies — default grounding, strict grounding, and citation-focused."
)
st.plotly_chart(plotly_bar({
    "V1 — Default":   [0.9016, 0.9356, 0.9058, 0.7996],
    "V2 — Strict":    [0.9058, 0.8041, 0.9275, 0.7865],
    "V3 — Citations": [0.8986, 0.8493, 0.8841, 0.7590],
}), use_container_width=True)
note_box("V2 (strict grounding) wins on Faithfulness + Context Recall but hurts Answer Relevancy. V2 is used in production.")

# --- 3. Reranker Comparison ---
section_header(
    "3 · Reranker Comparison (GPT-4o-mini)",
    "SBERT cross-encoder vs Cohere reranker — same retrieval, same model, only reranker differs."
)
st.plotly_chart(plotly_bar({
    "SBERT":  [0.9705, 0.9289, 0.9058, 0.8043],
    "Cohere": [0.8979, 0.9715, 0.9348, 0.8553],
}, y_min=0.85), use_container_width=True)
note_box("SBERT wins on Faithfulness by a large margin (0.971 vs 0.898). Cohere wins on Answer Relevancy. SBERT chosen as default.")

# --- 4. MLflow Experiment Log ---
st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin-bottom:24px;'>", unsafe_allow_html=True)
section_header(
    "4 · Systematic Experiment Log (MLflow — 11 Runs)",
    "11 experiments covering chunk size, overlap, reranker, model, rerank_n, and LangGraph. Hover to see all metrics per run."
)

# Dataframe
styled_exp = (
    df_exp.style
    .format("{:.4f}")
    .highlight_max(axis=0, props="font-weight:bold; color:#2ecc71")
    .highlight_min(axis=0, props="font-weight:bold; color:#e74c3c")
)
st.dataframe(styled_exp, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Metric filter
selected_metrics = st.multiselect(
    "Toggle metrics",
    options=METRICS,
    default=METRICS,
)

# Line chart with unified hover
fig2 = go.Figure()
runs = df_exp.index.tolist()

for metric, color in zip(METRICS, LINE_COLORS):
    visible = True if metric in selected_metrics else "legendonly"
    fig2.add_trace(go.Scatter(
        x=runs,
        y=df_exp[metric].values,
        mode="lines+markers",
        name=metric,
        line=dict(color=color, width=2.5),
        marker=dict(size=8, color=color, line=dict(width=1.5, color="rgba(255,255,255,0.3)")),
        hovertemplate=f"<b>{metric}</b>: %{{y:.4f}}<extra></extra>",
        visible=visible,
    ))

# Highlight best run (Run 03) — use numeric index
best_idx = runs.index("03 chunk=750/75")
fig2.add_vline(
    x=best_idx,
    line_width=1.5,
    line_dash="dash",
    line_color="rgba(240,192,96,0.5)",
    annotation_text="Best",
    annotation_font_color="rgba(240,192,96,0.8)",
    annotation_font_size=12,
)

line_layout = {**PLOT_LAYOUT}
line_layout["xaxis"] = dict(
    gridcolor="rgba(255,255,255,0.05)",
    tickfont=dict(color="rgba(200,195,225,0.75)", size=10),
    tickangle=-30,
    showline=True,
    linecolor="rgba(255,255,255,0.1)",
)
line_layout["yaxis"] = dict(
    gridcolor="rgba(255,255,255,0.05)",
    tickfont=dict(color="rgba(200,195,225,0.75)"),
    range=[0.45, 1.05],
    showline=True,
    linecolor="rgba(255,255,255,0.1)",
)
line_layout["margin"] = dict(l=50, r=20, t=30, b=120)
line_layout["hovermode"] = "x unified"
fig2.update_layout(**line_layout)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    "<div style='background:rgba(240,192,96,0.08); border-left:3px solid rgba(240,192,96,0.6); "
    "border-radius:4px; padding:12px 16px; font-size:14px; color:rgba(240,192,96,0.9); margin-top:4px;'>"
    "🏆 <strong>Key findings:</strong> chunk_size=750/75 is the overall winner — best faithfulness (0.9779) and "
    "answer relevancy (0.9748). Small chunks (200) hurt context precision badly (0.52). "
    "High overlap (150) hurts all metrics. LangGraph adds no measurable quality gain. "
    "Production config: chunk=750/75, SBERT, rerank_n=5, GPT-4o-mini."
    "</div>",
    unsafe_allow_html=True
)