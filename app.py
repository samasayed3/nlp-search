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

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Search Engine",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Intelligent Search Engine")
st.caption("Project 2 – NLP Course | Faculty of Computing & AI")

# ── Paths ───────────────────────────────────────────────────
REVIEWS_PATH = "data/Reviews.csv"
KAGGLE_DATASET = "snap/amazon-fine-food-reviews"

# ── Kaggle setup from Streamlit Secrets ────────────────────
def setup_kaggle_env():
    try:
        kaggle_user = st.secrets["KAGGLE_USERNAME"]
        kaggle_key = st.secrets["KAGGLE_KEY"]

        os.environ["KAGGLE_USERNAME"] = kaggle_user
        os.environ["KAGGLE_KEY"] = kaggle_key

    except Exception:
        st.error("❌ Kaggle credentials مش موجودة في Streamlit Secrets")
        st.stop()


def download_from_kaggle():
    import opendatasets as od

    st.info("📥 جاري تحميل الداتا من Kaggle...")

    od.download(
        f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
        data_dir="data"
    )

    src = "data/amazon-fine-food-reviews/Reviews.csv"
    if os.path.exists(src):
        os.makedirs("data", exist_ok=True)
        os.replace(src, REVIEWS_PATH)
        st.success("✅ تم تحميل الداتا بنجاح")
    else:
        st.error("❌ الملف غير موجود بعد التحميل")
        st.stop()


def ensure_data():
    if os.path.exists(REVIEWS_PATH):
        return

    st.warning("⚠️ الداتا مش موجودة — هيتم تحميلها تلقائيًا")

    setup_kaggle_env()
    download_from_kaggle()


# ── Ensure data exists ─────────────────────────────────────
ensure_data()


# ── Load models ────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading models...")
def load_models():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.data_loader import load_data, get_documents
    from src.preprocessing import preprocess_documents
    from src.tfidf_search import build_tfidf
    from src.embedding_search import build_embeddings

    df = load_data(REVIEWS_PATH, n_samples=3000)
    documents = get_documents(df)
    cleaned = preprocess_documents(documents)

    vectorizer, tfidf_matrix = build_tfidf(cleaned)
    emb_model, embeddings = build_embeddings(documents)

    return documents, vectorizer, tfidf_matrix, emb_model, embeddings


try:
    documents, vectorizer, tfidf_matrix, emb_model, embeddings = load_models()
    models_ready = True
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    models_ready = False


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("Top-K", 1, 10, 5)

    model_choice = st.radio(
        "Model",
        ["TF-IDF", "Embedding", "Both"],
        index=2
    )


# ── Input ──────────────────────────────────────────────────
query = st.text_input("Search:", key="q")

search = st.button("🔎 Search", disabled=not models_ready)


# ── Search ─────────────────────────────────────────────────
if search and query.strip():

    from src.tfidf_search import search_tfidf
    from src.embedding_search import search_embeddings

    st.markdown(f"### Results for: `{query}`")

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
        st.subheader("📄 TF-IDF")
        tfidf_results = search_tfidf(
            query, vectorizer, tfidf_matrix, documents, top_k
        )
        st.dataframe(to_df(tfidf_results))

    if model_choice in ["Embedding", "Both"]:
        st.subheader("🧠 Embedding")
        emb_results = search_embeddings(
            query, emb_model, embeddings, documents, top_k
        )
        st.dataframe(to_df(emb_results))


# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.caption("NLP Search Engine Project")
