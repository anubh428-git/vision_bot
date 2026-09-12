"""
scripts/build_index.py
-----------------------
One-time (or periodic) offline job: load the raw arXiv metadata dump, filter
to the configured domain categories, chunk the text, embed every chunk, and
persist a FAISS index + doc store to disk. The Streamlit app then just loads
these artifacts at startup instead of rebuilding the index every run.

Usage:
    # After downloading arxiv-metadata-oai-snapshot.json from Kaggle into data/:
    python scripts/build_index.py

    # Or point at a custom location / different category list:
    python scripts/build_index.py --json_path /path/to/arxiv.json --categories cs.LG cs.CL --max_papers 20000
"""
import argparse
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_pipeline import load_domain_subset, build_chunk_table
from src.vectorstore import ArxivVectorStore


def main():
    parser = argparse.ArgumentParser(description="Build the arXiv domain-expert chatbot's search index.")
    parser.add_argument("--json_path", default=config.RAW_ARXIV_JSON,
                         help="Path to arxiv-metadata-oai-snapshot.json (falls back to bundled sample if missing).")
    parser.add_argument("--categories", nargs="+", default=config.DOMAIN_CATEGORIES,
                         help="arXiv category codes to keep, e.g. cs.LG cs.CL cs.CV")
    parser.add_argument("--max_papers", type=int, default=config.MAX_PAPERS)
    args = parser.parse_args()

    print(f"[1/4] Loading + filtering records from {args.json_path} ...")
    df = load_domain_subset(json_path=args.json_path, categories=args.categories, max_papers=args.max_papers)
    print(f"      -> {len(df)} papers kept (categories: {args.categories})")

    if len(df) == 0:
        print("No papers matched -- check your json_path / categories and try again.")
        return

    print("[2/4] Chunking text ...")
    chunk_df = build_chunk_table(df)
    print(f"      -> {len(chunk_df)} chunks")

    print("[3/4] Embedding chunks and building FAISS index (this can take a while on the full dataset) ...")
    store = ArxivVectorStore()
    store.build(chunk_df)

    print("[4/4] Saving index + doc store to disk ...")
    store.save()
    df.to_parquet(config.PROCESSED_PARQUET, index=False)

    print("\nDone. Artifacts written to:")
    print(f"  {config.FAISS_INDEX_PATH}")
    print(f"  {config.DOC_STORE_PATH}")
    print(f"  {config.PROCESSED_PARQUET}")
    print("\nLaunch the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
