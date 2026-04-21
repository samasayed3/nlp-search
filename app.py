"""
app.py  –  Streamlit UI for the NLP Search Engine
===================================================
Run from the project root:
    streamlit run app.py
 
Input  : يكتب المستخدم query بالإنجليزي
Output : Top-5 نتايج من كل موديل (TF-IDF + Embedding) مع السكور
 
Data   : يتحمل تلقائياً من Kaggle أول ما يشتغل البرنامج
         (محتاج Kaggle username + API key في أول تشغيل)
"""
import os
import sys
import streamlit as st
import pandas as pd
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
# Paths
# ─────────────────────────────────────────────
REVIEWS_PATH = "data/Reviews.csv"
KAGGLE_DATASET = "snap/amazon-fine-food-reviews"

# ─────────────────────────────────────────────
# Kaggle setup from Streamlit secrets
# ─────────────────────────────────────────────
def setup_kaggle():
    """Setup Kaggle API credentials from Streamlit secrets"""
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
    st.info("📥 Downloading dataset from Kaggle...")

    api = KaggleApi()
    api.authenticate()

    os.makedirs("data", exist_ok=True)

    api.dataset_download_files(
        KAGGLE_DATASET,
        path="data",
        unzip=True
    )

    st.success("✅ Dataset downloaded successfully!")


def ensure_data():
    """Ensure dataset exists"""
    if os.path.exists(REVIEWS_PATH):
        return

    st.warning("⚠️ Dataset not found — downloading automatically...")

    try:
        setup_kaggle()
        download_dataset()
    except Exception as e:
        st.error(f"❌ Failed to download dataset: {e}")
        st.stop()


# ─────────────────────────────────────────────
# Ensure data exists
# ─────────────────────────────────────────────
ensure_data()


# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading models...")
def load_models():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.data_loader import load_data, get_documents
    from src.preprocessing import preprocess_documents
    from src.tfidf_search import build_tfidf
    from src.embedding_search import build_embeddings

    df = load_data(REVIEWS_PATH, n_samples=3000)
    documents = get_documents(df)
    cleaned_docs = preprocess_documents(documents)

    vectorizer, tfidf_matrix = build_tfidf(cleaned_docs)
    emb_model, embeddings = build_embeddings(documents)

    return documents, cleaned_docs, vectorizer, tfidf_matrix, emb_model, embeddings


# ─────────────────────────────────────────────
# Load models safely
# ─────────────────────────────────────────────
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
    st.header("⚙️ Settings")

    top_k = st.slider("Top-K results", 1, 10, 5)

    model_choice = st.radio(
        "Search Model",
        ["TF-IDF", "Embedding", "Both"],
        index=2
    )

    st.markdown("---")
    st.markdown("### Example Queries")

    examples = [
        "great coffee",
        "bad service",
        "healthy food",
        "delivery issue",
        "sweet cake"
    ]

    for q in examples:
        if st.button(q):
            st.session_state["query"] = q


# ─────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────
query = st.text_input(
    "Enter your query:",
    value=st.session_state.get("query", "")
)

search_btn = st.button("🔎 Search", disabled=not models_ready)


# ─────────────────────────────────────────────
# Search logic
# ─────────────────────────────────────────────
if search_btn and query.strip():

    from src.tfidf_search import search_tfidf
    from src.embedding_search import search_embeddings

    st.subheader(f"Results for: {query}")

    def to_df(results):
        return pd.DataFrame([
            {
                "Rank": r["rank"],
                "Score": r["score"],
                "Text": r["document"][:120] + "..."
            }
            for r in results
        ])

    if model_choice in ["TF-IDF", "Both"]:
        st.markdown("### TF-IDF Results")
        tfidf_results = search_tfidf(query, vectorizer, tfidf_matrix, documents, top_k)
        st.dataframe(to_df(tfidf_results), use_container_width=True)

    if model_choice in ["Embedding", "Both"]:
        st.markdown("### Embedding Results")
        emb_results = search_embeddings(query, emb_model, embeddings, documents, top_k)
        st.dataframe(to_df(emb_results), use_container_width=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Team: Sama & Team | NLP Project 2")
