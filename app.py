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
 
# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Search Engine",
    page_icon="🔍",
    layout="wide",
)
 
st.title("🔍 Intelligent Search Engine")
st.caption("Project 2 – NLP Course | Faculty of Computing & AI")
 
# ── Kaggle credentials (يظهر بس لو الداتا مش موجودة) ─────────────────────────
REVIEWS_PATH = "data/Reviews.csv"
KAGGLE_DATASET = "snap/amazon-fine-food-reviews"
 
def download_from_kaggle():
    """تحميل الداتا من Kaggle باستخدام opendatasets."""
    try:
        import opendatasets as od
    except ImportError:
        st.error("❌ مكتبة `opendatasets` مش موجودة. شغّل: `pip install opendatasets`")
        st.stop()
 
    st.info("📥 جارٍ تحميل الداتا من Kaggle...")
    try:
        # opendatasets بيحمّل في مجلد باسم الداتاست
        od.download(f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}", data_dir="data")
        # بينزل في data/amazon-fine-food-reviews/Reviews.csv
        downloaded_path = "data/amazon-fine-food-reviews/Reviews.csv"
        if os.path.exists(downloaded_path):
            os.makedirs("data", exist_ok=True)
            import shutil
            shutil.copy(downloaded_path, REVIEWS_PATH)
            st.success("✅ تم تحميل الداتا بنجاح!")
        else:
            st.error(f"❌ مش لاقي الملف في {downloaded_path} – تأكد من الـ credentials.")
            st.stop()
    except Exception as e:
        st.error(f"❌ فشل التحميل: {e}")
        st.stop()
 
 
def ensure_data_available():
    """تأكد إن الداتا موجودة، لو لأ اطلب credentials وحمّل."""
    if os.path.exists(REVIEWS_PATH):
        return  # الداتا موجودة، مفيش حاجة
 
    st.warning("⚠️ ملف البيانات مش موجود. هيتحمّل من Kaggle دلوقتي.")
    st.markdown(
        "محتاج **Kaggle API credentials** (username + key).  \n"
        "تقدر تجيبهم من: [kaggle.com/settings](https://www.kaggle.com/settings) ← API ← Create New Token"
    )
 
    with st.form("kaggle_credentials"):
        kaggle_user = st.text_input("Kaggle Username")
        kaggle_key  = st.text_input("Kaggle API Key", type="password")
        submitted   = st.form_submit_button("⬇️ حمّل الداتا")
 
    if submitted:
        if not kaggle_user or not kaggle_key:
            st.error("❌ حط الـ username والـ key الأول.")
            st.stop()
 
        # opendatasets بيقرأ credentials من ~/.kaggle/kaggle.json أو من stdin
        # بنعمل الملف يدوياً عشان نتجنب الـ interactive prompt
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        import json
        with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
            json.dump({"username": kaggle_user, "key": kaggle_key}, f)
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
 
        download_from_kaggle()
        st.rerun()
    else:
        st.stop()
 
 
# ── تأكد من وجود الداتا قبل أي حاجة تانية ────────────────────────────────────
ensure_data_available()
 
 
# ── Load models (cached so they don't reload every time) ──────────────────────
@st.cache_resource(show_spinner="⏳ جارٍ تحميل الموديلات – انتظر لحظة...")
def load_models():
    """Load data, build TF-IDF and Embedding models – runs once."""
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
 
 
# ── Try loading ────────────────────────────────────────────────────────────────
try:
    documents, cleaned_docs, vectorizer, tfidf_matrix, emb_model, embeddings = load_models()
    models_ready = True
except Exception as e:
    st.error(f"❌ حصل خطأ أثناء تحميل الموديلات: {e}")
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
    "اكتب الـ Query بتاعك هنا:",
    value=st.session_state.get("query_input", ""),
    placeholder="مثلاً: food delivery problem",
    key="query_input",
)
 
search_clicked = st.button("🔎 ابحث", type="primary", disabled=not models_ready)
 
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
    st.warning("⚠️ اكتب query الأول!")
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "👩‍💻 Team: Gehad · Alaa · Waad · Aliaa · Sama · Aya  |  "
    "Dataset: Amazon Fine Food Reviews"
)
 
