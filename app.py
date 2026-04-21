"""
app.py  –  Streamlit UI for the NLP Search Engine
===================================================
Run from the project root:
    streamlit run app.py

Input  : يكتب المستخدم query بالإنجليزي
Output : Top-5 نتايج من كل موديل (TF-IDF + Embedding) مع السكور
"""

import streamlit as st
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Search Engine",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Intelligent Search Engine")
st.caption("Project 2 – NLP Course | Faculty of Computing & AI")

# ── Load models (cached so they don't reload every time) ──────────────────────
@st.cache_resource(show_spinner="⏳ جارٍ تحميل الموديلات – انتظري لحظة...")
def load_models():
    """Load data, build TF-IDF and Embedding models – runs once."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.data_loader      import load_data, get_documents
    from src.preprocessing    import preprocess_documents
    from src.tfidf_search     import build_tfidf
    from src.embedding_search import build_embeddings

    df           = load_data(r"C:\Users\Lenovo\Downloads\nlp-search-engine-main (1)\nlp-search-engine-main\data\Reviews.csv", n_samples=3000)
    documents    = get_documents(df)
    cleaned_docs = preprocess_documents(documents)
    vectorizer, tfidf_matrix = build_tfidf(cleaned_docs)
    emb_model, embeddings    = build_embeddings(documents)

    return documents, cleaned_docs, vectorizer, tfidf_matrix, emb_model, embeddings


# ── Try loading – show friendly error if CSV missing ─────────────────────────
try:
    documents, cleaned_docs, vectorizer, tfidf_matrix, emb_model, embeddings = load_models()
    models_ready = True
except FileNotFoundError:
    st.error(
        "❌ ملف البيانات مش موجود.\n\n"
        "حملي **Reviews.csv** من Kaggle وحطيه في مجلد `data/`\n\n"
        "🔗 https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews"
    )
    models_ready = False

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ الإعدادات")
    top_k = st.slider("عدد النتايج (Top-K)", min_value=1, max_value=10, value=5)
    model_choice = st.radio(
        "الموديل",
        ["TF-IDF فقط", "Embedding فقط", "المقارنة (الاتنين)"],
        index=2,
    )
    st.markdown("---")
    st.markdown("**مثال على queries:**")
    example_queries = [
        "great coffee and pastries",
        "bad service and cold food",
        "healthy snacks for kids",
        "food delivery problem",
        "sweet chocolate cake",
    ]
    for q in example_queries:
        if st.button(q, use_container_width=True):
            st.session_state["query_input"] = q

# ── Query input ───────────────────────────────────────────────────────────────
query = st.text_input(
    "اكتبي الـ Query بتاعك هنا:",
    value=st.session_state.get("query_input", ""),
    placeholder="مثلاً: food delivery problem",
    key="query_input",
)

search_clicked = st.button("🔎 ابحثي", type="primary", disabled=not models_ready)

# ── Search ────────────────────────────────────────────────────────────────────
if search_clicked and query.strip():
    from src.tfidf_search     import search_tfidf
    from src.embedding_search import search_embeddings

    st.markdown(f"### نتايج البحث عن: `{query}`")

    def results_to_df(results):
        return pd.DataFrame([
            {
                "Rank": r["rank"],
                "Score": r["score"],
                "Document (first 120 chars)": r["document"][:120] + "...",
            }
            for r in results
        ])

    show_tfidf = model_choice in ["TF-IDF فقط", "المقارنة (الاتنين)"]
    show_emb   = model_choice in ["Embedding فقط", "المقارنة (الاتنين)"]

    if show_tfidf and show_emb:
        col1, col2 = st.columns(2)
    elif show_tfidf:
        col1 = st.container()
        col2 = None
    else:
        col1 = None
        col2 = st.container()

    if show_tfidf:
        with col1:
            st.subheader("📄 TF-IDF (Baseline)")
            tfidf_results = search_tfidf(
                query, vectorizer, tfidf_matrix, documents, top_k=top_k
            )
            st.dataframe(results_to_df(tfidf_results), use_container_width=True, hide_index=True)

    if show_emb:
        with col2:
            st.subheader("🧠 Embedding (Advanced)")
            emb_results = search_embeddings(
                query, emb_model, embeddings, documents, top_k=top_k
            )
            st.dataframe(results_to_df(emb_results), use_container_width=True, hide_index=True)

    # ── Full document expanders ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📖 النص الكامل للنتايج")

    all_results = []
    if show_tfidf:
        for r in tfidf_results:
            all_results.append(("TF-IDF", r))
    if show_emb:
        for r in emb_results:
            all_results.append(("Embedding", r))

    for model_name, r in all_results:
        with st.expander(f"[{model_name}] Rank #{r['rank']} – Score {r['score']}"):
            st.write(r["document"])

elif search_clicked and not query.strip():
    st.warning("⚠️ اكتبي query الأول!")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "👩‍💻 Team: Gehad · Alaa · Waad · Aliaa · Sama · Aya  |  "
    "Dataset: Amazon Fine Food Reviews"
)
