"""
app.py  –  Streamlit UI for the NLP Search Engine
===================================================
Run from the project root:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from kaggle.api.kaggle_api_extended import KaggleApi

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Search Engine",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Intelligent Search Engine")
st.caption("Project 2 – NLP Course | Faculty of Computing & AI")

# ─────────────────────────────────────────────
# Data path (Kaggle auto download)
# ─────────────────────────────────────────────
REVIEWS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Reviews.csv")
KAGGLE_DATASET = "snap/amazon-fine-food-reviews"


def setup_kaggle():
    """Setup Kaggle credentials from Streamlit secrets"""
    try:
        kaggle_user = st.secrets["KAGGLE_USERNAME"]
        kaggle_key = st.secrets["KAGGLE_KEY"]
    except Exception:
        st.error("❌ لازم تضيف Kaggle credentials في Streamlit secrets")
        st.stop()

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
        json.dump({"username": kaggle_user, "key": kaggle_key}, f)

    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)


def download_dataset():
    """Download dataset using Kaggle API"""
    st.info("📥 جارٍ تحميل الداتا من Kaggle...")

    api = KaggleApi()
    api.authenticate()

    os.makedirs("data", exist_ok=True)

    api.dataset_download_files(
        KAGGLE_DATASET,
        path="data",
        unzip=True
    )

    # نقل الملف للمسار المطلوب
    downloaded_path = os.path.join("data", "amazon-fine-food-reviews", "Reviews.csv")
    if os.path.exists(downloaded_path):
        import shutil
        shutil.copy(downloaded_path, REVIEWS_PATH)

    st.success("✅ تم تحميل الداتا بنجاح!")


def ensure_data():
    """Ensure dataset exists"""
    if os.path.exists(REVIEWS_PATH):
        return

    st.warning("⚠️ الداتا مش موجودة — هيتم تحميلها تلقائيًا")

    try:
        setup_kaggle()
        download_dataset()
    except Exception as e:
        st.error(f"❌ فشل تحميل الداتا: {e}")
        st.stop()


# ─────────────────────────────────────────────
# Ensure data exists
# ─────────────────────────────────────────────
ensure_data()


# ─────────────────────────────────────────────
# Load models  (cached – runs once per session)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ جارٍ تحميل الموديلات – انتظر لحظة...")
def load_models():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.data_loader      import load_data, get_documents
    from src.preprocessing    import preprocess_documents
    from src.tfidf_search     import build_tfidf
    from src.embedding_search import build_embeddings

    df           = load_data(REVIEWS_PATH, n_samples=3000)
    documents    = get_documents(df)
    cleaned_docs = preprocess_documents(documents)

    vectorizer, tfidf_matrix = build_tfidf(cleaned_docs)
    emb_model, embeddings    = build_embeddings(documents)

    return documents, cleaned_docs, vectorizer, tfidf_matrix, emb_model, embeddings


try:
    documents, cleaned_docs, vectorizer, tfidf_matrix, emb_model, embeddings = load_models()
    models_ready = True
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    models_ready = False


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ الإعدادات")
    top_k = st.slider("عدد النتايج (Top-K)", 1, 10, 5)

    model_choice = st.radio(
        "الموديل",
        ["TF-IDF فقط", "Embedding فقط", "مقارنة الموديلات", "الكل"],
        index=2,
    )

    st.markdown("---")
    st.markdown("### مثال على queries")
    examples = [
        "great coffee and pastries",
        "bad service and cold food",
        "healthy snacks for kids",
        "delivery issue",
        "sweet chocolate cake",
    ]
    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state["query"] = q


# ─────────────────────────────────────────────
# Query input
# ─────────────────────────────────────────────
query = st.text_input(
    "اكتب الـ Query بتاعك هنا:",
    value=st.session_state.get("query", ""),
    placeholder="مثلاً: great coffee and pastries",
    key="query",
)

search_btn = st.button("🔎 ابحث", type="primary", disabled=not models_ready)


# ─────────────────────────────────────────────
# Search  +  Evaluation
# ─────────────────────────────────────────────
if search_btn and query.strip():

    from src.tfidf_search      import search_tfidf
    from src.embedding_search  import search_embeddings
    from src.evaluation        import evaluate_for_streamlit

    st.subheader(f"نتايج البحث عن: `{query}`")

    tfidf_results = search_tfidf(query, vectorizer, tfidf_matrix, documents, top_k)
    emb_results   = search_embeddings(query, emb_model, embeddings, documents, top_k)

    def to_df(results):
        return pd.DataFrame([
            {"Rank": r["rank"], "Score": f"{r['score']:.4f}",
             "Document (first 120 chars)": r["document"][:120] + "…"}
            for r in results
        ])

    show_tfidf = model_choice in ["TF-IDF فقط", "الكل"]
    show_emb   = model_choice in ["Embedding فقط", "الكل"]
    show_eval  = model_choice in ["مقارنة الموديلات", "الكل"]

    if show_tfidf:
        st.markdown("### 📄 TF-IDF Results")
        st.dataframe(to_df(tfidf_results), use_container_width=True, hide_index=True)

    if show_emb:
        st.markdown("### 🧠 Embedding Results")
        st.dataframe(to_df(emb_results), use_container_width=True, hide_index=True)

    if show_eval:
        st.markdown("---")
        st.markdown("### 📊 تقييم الموديلات – Model Evaluation")

        eval_data = evaluate_for_streamlit(query, tfidf_results, emb_results, k=top_k)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"TF-IDF Precision@{top_k}", f"{eval_data['p_tfidf']:.4f}")
        with col2:
            st.metric(f"Embedding Precision@{top_k}", f"{eval_data['p_emb']:.4f}")
        with col3:
            icons = {"TF-IDF": "🏆 TF-IDF", "Embedding": "🏆 Embedding", "Tie": "🤝 Tie"}
            st.metric("Winner", icons[eval_data["winner"]])

        st.markdown("#### Precision Comparison")
        st.pyplot(eval_data["bar_fig"])
        plt.close(eval_data["bar_fig"])

        st.markdown("#### Summary")
        st.dataframe(
            pd.DataFrame(eval_data["summary_rows"]),
            use_container_width=True, hide_index=True
        )

        st.markdown("#### Detailed Results Comparison")
        st.dataframe(
            pd.DataFrame(eval_data["detail_rows"]),
            use_container_width=True, hide_index=True
        )

        st.markdown("#### Relevance Statistics")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**TF-IDF**")
            st.write(f"- Relevant in top-{top_k}: {sum(eval_data['tfidf_flags'])}/{top_k}")
            st.write(f"- Precision@{top_k}: {eval_data['p_tfidf']:.4f}")
        with c2:
            st.markdown("**Embedding**")
            st.write(f"- Relevant in top-{top_k}: {sum(eval_data['emb_flags'])}/{top_k}")
            st.write(f"- Precision@{top_k}: {eval_data['p_emb']:.4f}")

    st.markdown("---")
    st.markdown("#### 📖 النص الكامل للنتايج")

    all_results = []
    if show_tfidf or show_eval:
        for r in tfidf_results:
            all_results.append(("TF-IDF", r))
    if show_emb or show_eval:
        for r in emb_results:
            all_results.append(("Embedding", r))

    for model_name, r in all_results:
        with st.expander(f"[{model_name}] Rank #{r['rank']} – Score {r['score']:.4f}"):
            st.write(r["document"])

elif search_btn and not query.strip():
    st.warning("⚠️ اكتب query الأول!")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("👩‍💻 Team: Gehad · Alaa · Waad · Aliaa · Sama · Aya  |  Dataset: Amazon Fine Food Reviews")
