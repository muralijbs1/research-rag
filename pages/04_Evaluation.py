from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Evaluation Results", page_icon="📊", layout="wide")
st.title("RAGAS Evaluation Results")
st.caption(
    "Static results from one-time RAGAS runs over 23 questions across 11 papers. "
    "Metrics are averaged across all questions."
)

METRICS = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"]
COLORS = ["#4A90D9", "#E67E22", "#2ECC71", "#9B59B6"]


def grouped_bar_chart(data: dict, y_min: float = 0.0):
    labels = list(data.keys())
    values = np.array(list(data.values()))  # (n_variants, 4)

    n_metrics = len(METRICS)
    n_variants = len(labels)
    x = np.arange(n_metrics)
    total_width = 0.7
    bar_width = total_width / n_variants
    offsets = np.linspace(
        -total_width / 2 + bar_width / 2,
         total_width / 2 - bar_width / 2,
        n_variants,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (label, offset) in enumerate(zip(labels, offsets)):
        bars = ax.bar(
            x + offset, values[i], width=bar_width,
            label=label, color=COLORS[i % len(COLORS)], zorder=3,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=10)
    ax.set_ylim(y_min, 1.04)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def show_section(title: str, data: dict, note: str, y_min: float = 0.0):
    st.subheader(title)

    df = pd.DataFrame(data, index=METRICS).T
    df.index.name = "Model / Variant"
    styled = (
        df.style
        .format("{:.4f}")
        .highlight_max(axis=0, props="font-weight:bold; color:#2ecc71")
    )
    st.dataframe(styled, use_container_width=True)

    fig = grouped_bar_chart(data, y_min=y_min)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(note)
    st.divider()


# ------------------------------------------------------------------
# 1. Model comparison  (report_ragas_baseline.ipynb)
# ------------------------------------------------------------------
show_section(
    "1 · Model Comparison",
    {
        "GPT-4o-mini":    [0.9016, 0.9356, 0.9058, 0.7996],
        "GPT-4o-mini v2": [0.9040, 0.7618, 0.9130, 0.7776],
        "Claude (Haiku)": [0.9114, 0.7282, 0.9130, 0.8093],
    },
    "GPT-4o-mini leads on Answer Relevancy. Claude edges ahead on Faithfulness and Context Precision.",
)

# ------------------------------------------------------------------
# 2. Prompt variants  (scratch_prompt_lab.ipynb — all GPT-4o-mini)
# ------------------------------------------------------------------
show_section(
    "2 · Prompt Variants  (GPT-4o-mini)",
    {
        "V1 — Default":   [0.9016, 0.9356, 0.9058, 0.7996],
        "V2 — Strict":    [0.9058, 0.8041, 0.9275, 0.7865],
        "V3 — Citations": [0.8986, 0.8493, 0.8841, 0.7590],
    },
    (
        "V2 (strict grounding) wins on Faithfulness + Context Recall but hurts Answer Relevancy — "
        "the model becomes too conservative. V3 (citation instructions) adds overhead without gain. "
        "V2 is the best overall trade-off."
    ),
)

# ------------------------------------------------------------------
# 3. Reranker comparison  (report_reranker_comparison.ipynb — GPT-4o-mini)
# ------------------------------------------------------------------
show_section(
    "3 · Reranker Comparison  (GPT-4o-mini)",
    {
        "SBERT":  [0.9705, 0.9289, 0.9058, 0.8043],
        "Cohere": [0.8979, 0.9715, 0.9348, 0.8553],
    },
    (
        "SBERT wins on Faithfulness by a large margin (0.971 vs 0.898). "
        "Cohere wins on Answer Relevancy and Context Precision. "
        "SBERT chosen as default — faithfulness is the higher-priority metric."
    ),
)
