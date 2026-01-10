"""
Embedding Similarity (TF-IDF based)

Purpose:
- Compute TF-IDF based similarity scores (lightweight alternative to sentence transformers)
- Compare header-only vs header+context representations
"""

import os
import argparse
import pandas as pd
import yaml
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase D2: Embedding Similarity")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]

    input_path = os.path.join(tables_dir, "schema_pairs_repr.csv")
    df = pd.read_csv(input_path)

    # Use TF-IDF vectorizer for lightweight embeddings
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    
    # Encode header-only texts
    left_header_texts = df["left_header_only"].astype(str).tolist()
    right_header_texts = df["right_header_only"].astype(str).tolist()
    all_header_texts = left_header_texts + right_header_texts
    
    vectorizer.fit(all_header_texts)
    left_header_emb = vectorizer.transform(left_header_texts)
    right_header_emb = vectorizer.transform(right_header_texts)

    # Encode header+context texts
    left_ctx_texts = df["left_header_with_context"].astype(str).tolist()
    right_ctx_texts = df["right_header_with_context"].astype(str).tolist()
    all_ctx_texts = left_ctx_texts + right_ctx_texts
    
    vectorizer_ctx = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    vectorizer_ctx.fit(all_ctx_texts)
    left_ctx_emb = vectorizer_ctx.transform(left_ctx_texts)
    right_ctx_emb = vectorizer_ctx.transform(right_ctx_texts)

    # Compute cosine similarities row-wise
    header_only_scores = []
    header_ctx_scores = []

    for i in range(len(df)):
        header_only_scores.append(
            cosine_similarity(
                left_header_emb[i].reshape(1, -1),
                right_header_emb[i].reshape(1, -1)
            )[0][0]
        )
        header_ctx_scores.append(
            cosine_similarity(
                left_ctx_emb[i].reshape(1, -1),
                right_ctx_emb[i].reshape(1, -1)
            )[0][0]
        )

    out_df = df[
        ["left_city", "right_city", "left_column_final", "right_column_final"]
    ].copy()

    out_df["embedding_header_only_cosine"] = header_only_scores
    out_df["embedding_header_context_cosine"] = header_ctx_scores

    output_path = os.path.join(tables_dir, "similarity_scores_embedding.csv")
    out_df.to_csv(output_path, index=False)

    print(f"[OK] Embedding similarity scores saved: {output_path}")
    print(f"[INFO] Total rows: {len(out_df)}")


if __name__ == "__main__":
    main()