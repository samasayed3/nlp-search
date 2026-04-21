"""
main_full.py  –  NLP Search Engine (Unified Entry Point)
=========================================================
Project 2: Intelligent Search Engine
Faculty of Computing & Artificial Intelligence – Spring 2025-2026

This single file orchestrates ALL project components by importing from the
existing src/ modules. Run it from the project root:

    python main_full.py

Team:
    Gehad  – Data Loader
    Alaa   – Preprocessing
    Waad   – TF-IDF Baseline
    Aliaa  – Embedding Advanced Search
    Sama   – Evaluation
    Aya    – Report / Demo Notebook

NOTE: Make sure   data/Reviews.csv   exists before running.
Download from: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS  (all from the original src/ modules – nothing duplicated here)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

# Add project root to path so  src.*  imports always resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader      import load_data, get_documents, save_sample      # Gehad
from src.preprocessing    import preprocess_documents                        # Alaa
from src.tfidf_search     import build_tfidf, search_tfidf                  # Waad
from src.embedding_search import build_embeddings, search_embeddings         # Aliaa
from src.evaluation       import evaluate                                    # Sama


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  –  change these to tune the system
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH   = "data/Reviews.csv"   # path to the CSV dataset
N_SAMPLES   = 3000                 # number of reviews to load
TOP_K       = 5                    # how many results to show per query

# Test queries used for both demonstration and evaluation
QUERIES = [
    "great coffee and pastries",
    "bad service and cold food",
    "healthy snacks for kids",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str):
    """Print a formatted section banner."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _print_results(results: list, top_k: int = TOP_K):
    """Pretty-print a list of search result dicts."""
    for r in results[:top_k]:
        preview = r["document"][:90].replace("\n", " ")
        print(f"    #{r['rank']}  score={r['score']:.4f}  |  {preview}...")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── Step 1 · Load data ───────────────────────────────────────────────────
    _banner("STEP 1 – Loading data")
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at '{DATA_PATH}'.")
        print("        Download Reviews.csv from Kaggle and place it in data/")
        sys.exit(1)

    df        = load_data(DATA_PATH, n_samples=N_SAMPLES)   # Gehad's function
    documents = get_documents(df)                           # Gehad's function
    save_sample(df)                                         # Gehad's function – saves data/sample_500.csv
    print(f"  Loaded {len(documents):,} reviews from '{DATA_PATH}'")


    # ── Step 2 · Preprocess ──────────────────────────────────────────────────
    _banner("STEP 2 – Preprocessing (Alaa)")
    cleaned_docs = preprocess_documents(documents)          # Alaa's function
    print(f"  Preprocessed {len(cleaned_docs):,} documents  ✓")

    # Quick sanity check: show one example
    print(f"\n  Example – original  : {documents[0][:80]}...")
    print(f"  Example – cleaned   : {cleaned_docs[0][:80]}...")


    # ── Step 3 · TF-IDF model ────────────────────────────────────────────────
    _banner("STEP 3 – Building TF-IDF model (Waad)")
    vectorizer, tfidf_matrix = build_tfidf(cleaned_docs)    # Waad's function


    # ── Step 4 · Embedding model ─────────────────────────────────────────────
    _banner("STEP 4 – Building Embedding model (Aliaa)")
    emb_model, embeddings = build_embeddings(documents)     # Aliaa's function


    # ── Step 5 · Demo search ─────────────────────────────────────────────────
    _banner("STEP 5 – Demo search results")
    for query in QUERIES:
        print(f"\n  Query: \"{query}\"")

        print(f"  · TF-IDF top-{TOP_K}:")
        tfidf_res = search_tfidf(                           # Waad's function
            query, vectorizer, tfidf_matrix, documents, top_k=TOP_K
        )
        _print_results(tfidf_res)

        print(f"\n  · Embedding top-{TOP_K}:")
        emb_res = search_embeddings(                        # Aliaa's function
            query, emb_model, embeddings, documents, top_k=TOP_K
        )
        _print_results(emb_res)

        print()


    # ── Step 6 · Evaluation ──────────────────────────────────────────────────
    _banner("STEP 6 – Evaluation  Precision@k  (Sama)")
    evaluate(                                               # Sama's function
        queries      = QUERIES,
        vectorizer   = vectorizer,
        tfidf_matrix = tfidf_matrix,
        emb_model    = emb_model,
        embeddings   = embeddings,
        cleaned_docs = cleaned_docs,
        documents    = documents,
        k            = TOP_K,
    )

    _banner("DONE")
    print("  evaluation_results.png  →  saved in project root")
    print("  data/sample_500.csv     →  saved by data_loader")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
