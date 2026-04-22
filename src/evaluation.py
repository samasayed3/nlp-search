"""
evaluation.py
Member 5 – Sama
Precision@k evaluation + comparison tables and charts for both models.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from src.tfidf_search import search_tfidf
    from src.embedding_search import search_embeddings
except ModuleNotFoundError:
    from tfidf_search import search_tfidf
    from embedding_search import search_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE ORACLE
# ─────────────────────────────────────────────────────────────────────────────

QUERY_KEYWORDS = {
    "great coffee and pastries"  : ["coffee", "pastri", "cafe", "espresso", "latte", "muffin", "donut", "biscuit", "croissant", "baked"],
    "bad service and cold food"  : ["bad", "cold", "terrible", "awful", "horrible", "disappoint", "poor", "worst", "rude", "slow"],
    "healthy snacks for kids"    : ["healthy", "kid", "child", "snack", "organic", "natural", "wholesome", "nutritious", "fruit", "veggie"],
}


def _is_relevant(document: str, keywords: list) -> int:
    """Return 1 if the document contains at least one keyword, else 0."""
    doc_lower = document.lower()
    return int(any(kw in doc_lower for kw in keywords))


def _build_relevant_flags(query: str, results: list) -> list:
    if query in QUERY_KEYWORDS:
        keywords = QUERY_KEYWORDS[query]
    else:
        keywords = [w.lower() for w in query.split() if len(w) > 3]
    return [_is_relevant(r["document"], keywords) for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  precision_at_k
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(relevant_flags: list, k: int) -> float:
    if k <= 0:
        return 0.0
    top_k_flags = relevant_flags[:k]
    return round(sum(top_k_flags) / k, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  evaluate_model  (no printing – returns float only)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model_name: str, results: list, relevant_flags: list, k: int = 5) -> float:
    return precision_at_k(relevant_flags, k)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  evaluate_for_streamlit  – single query, returns figures + table data
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_for_streamlit(query: str, tfidf_results: list, emb_results: list, k: int = 5):
    """
    Evaluate a single query for both models.
    Returns a dict with precision scores, relevance flags, a bar chart Figure,
    summary rows, and detailed per-rank rows – ready for Streamlit display.
    """
    tfidf_flags = _build_relevant_flags(query, tfidf_results)
    emb_flags   = _build_relevant_flags(query, emb_results)

    p_tfidf = precision_at_k(tfidf_flags, k)
    p_emb   = precision_at_k(emb_flags, k)

    if p_tfidf > p_emb:
        winner = "TF-IDF"
    elif p_emb > p_tfidf:
        winner = "Embedding"
    else:
        winner = "Tie"

    bar_fig = _plot_single_query(query, p_tfidf, p_emb, k)

    summary_rows = [
        {
            "Metric"   : f"Precision@{k}",
            "TF-IDF"   : f"{p_tfidf:.4f}",
            "Embedding": f"{p_emb:.4f}",
            "Winner"   : winner,
        },
        {
            "Metric"   : "Relevant Docs",
            "TF-IDF"   : f"{sum(tfidf_flags[:k])}/{k}",
            "Embedding": f"{sum(emb_flags[:k])}/{k}",
            "Winner"   : "",
        },
    ]

    detail_rows = []
    for i in range(min(k, len(tfidf_results), len(emb_results))):
        detail_rows.append({
            "Rank"              : i + 1,
            "TF-IDF Score"      : round(tfidf_results[i]["score"], 4),
            "TF-IDF Relevant"   : "✅" if tfidf_flags[i] == 1 else "❌",
            "TF-IDF Doc"        : tfidf_results[i]["document"][:100] + "…",
            "Embedding Score"   : round(emb_results[i]["score"], 4),
            "Embedding Relevant": "✅" if emb_flags[i] == 1 else "❌",
            "Embedding Doc"     : emb_results[i]["document"][:100] + "…",
        })

    return {
        "p_tfidf"     : p_tfidf,
        "p_emb"       : p_emb,
        "tfidf_flags" : tfidf_flags[:k],
        "emb_flags"   : emb_flags[:k],
        "winner"      : winner,
        "bar_fig"     : bar_fig,
        "summary_rows": summary_rows,
        "detail_rows" : detail_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  evaluate  – multi-query, returns figures + table data (no print / no savefig)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(queries, vectorizer, tfidf_matrix, emb_model, embeddings,
             cleaned_docs, documents, k=5):
    """
    Run Precision@k for both models across all queries.
    Returns a dict with a comparison Figure, per-query table rows, and averages.
    """
    tfidf_scores, emb_scores = [], []

    for query in queries:
        tfidf_results = search_tfidf(query, vectorizer, tfidf_matrix, documents, top_k=k)
        p_tfidf = precision_at_k(_build_relevant_flags(query, tfidf_results), k)

        emb_results = search_embeddings(query, emb_model, embeddings, documents, top_k=k)
        p_emb = precision_at_k(_build_relevant_flags(query, emb_results), k)

        tfidf_scores.append(p_tfidf)
        emb_scores.append(p_emb)

    avg_tfidf = float(np.mean(tfidf_scores))
    avg_emb   = float(np.mean(emb_scores))
    overall_winner = "TF-IDF" if avg_tfidf > avg_emb else ("Embedding" if avg_emb > avg_tfidf else "Tie")

    table_rows = []
    for query, p_tf, p_emb in zip(queries, tfidf_scores, emb_scores):
        w = "TF-IDF" if p_tf > p_emb else ("Embedding" if p_emb > p_tf else "Tie")
        table_rows.append({"Query": query, "TF-IDF": round(p_tf, 4),
                           "Embedding": round(p_emb, 4), "Winner": w})

    return {
        "comparison_fig": _plot_comparison(queries, tfidf_scores, emb_scores, k),
        "table_rows"    : table_rows,
        "avg_tfidf"     : avg_tfidf,
        "avg_emb"       : avg_emb,
        "overall_winner": overall_winner,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_single_query(query: str, p_tfidf: float, p_emb: float, k: int):
    """Bar chart for one query. Returns Figure."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["TF-IDF", "Embedding"]
    scores = [p_tfidf, p_emb]
    colors = ["#4C72B0", "#DD8452"]

    bars = ax.bar(models, scores, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    ax.set_ylabel(f"Precision@{k}", fontsize=12)
    short_q = (query[:47] + "…") if len(query) > 50 else query
    ax.set_title(f"Model Comparison — '{short_q}'", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.4f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    winner_idx = 0 if p_tfidf >= p_emb else 1
    bars[winner_idx].set_edgecolor("gold")
    bars[winner_idx].set_linewidth(3)
    plt.tight_layout()
    return fig


def _plot_comparison(queries: list, tfidf_scores: list, emb_scores: list, k: int):
    """Grouped + average bar chart for multiple queries. Returns Figure."""
    short_queries = [q[:30] + "…" if len(q) > 30 else q for q in queries]
    x     = np.arange(len(queries))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Search Engine Evaluation  —  Precision@{k}", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    bars1 = ax1.bar(x - width / 2, tfidf_scores, width, label="TF-IDF",
                    color="#4C72B0", edgecolor="white", linewidth=0.8)
    bars2 = ax1.bar(x + width / 2, emb_scores,   width, label="Embedding",
                    color="#DD8452", edgecolor="white", linewidth=0.8)
    ax1.set_xlabel("Query")
    ax1.set_ylabel(f"Precision@{k}")
    ax1.set_title(f"Per-Query Precision@{k}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_queries, rotation=20, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.legend()
    ax1.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax1.set_axisbelow(True)
    for bar in [*bars1, *bars2]:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    avg_tf  = float(np.mean(tfidf_scores))
    avg_emb = float(np.mean(emb_scores))
    bars = ax2.bar(["TF-IDF", "Embedding"], [avg_tf, avg_emb],
                   color=["#4C72B0", "#DD8452"], edgecolor="white", linewidth=0.8, width=0.4)
    ax2.set_ylabel(f"Average Precision@{k}")
    ax2.set_title(f"Overall Average Precision@{k}")
    ax2.set_ylim(0, 1.15)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax2.set_axisbelow(True)
    for bar, val in zip(bars, [avg_tf, avg_emb]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    winner_idx = 0 if avg_tf >= avg_emb else 1
    bars[winner_idx].set_edgecolor("gold")
    bars[winner_idx].set_linewidth(2.5)
    winner_patch = mpatches.Patch(edgecolor="gold", facecolor="none",
                                  linewidth=2.5, label="Winner")
    ax2.legend(handles=[winner_patch])
    plt.tight_layout()
    return fig
