from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "malayalam_df_clean.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "tfidf"
)

RESULTS_PATH = RESULTS_DIR / "tfidf_results.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_config import QUERIES, TOP_K

TEXT_COLUMN = "clean_synopsis"

EXPECTED_ROW_COUNT = 94

TFIDF_CONFIG = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_df": 0.95,
    "sublinear_tf": True,
    "norm": "l2",
    "lowercase": False,
    "strip_accents": None,
    "stop_words": None,
}

def load_and_validate_corpus(data_path=DATA_PATH):
    df = pd.read_csv(data_path)

    required_columns = {
        "source_row_id",
        "title",
        "synopsis",
        "genre",
        "rating",
        "movie_language",
        "title_confidence",
        TEXT_COLUMN,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if len(df) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} records, "
            f"but found {len(df)}."
        )

    if df[TEXT_COLUMN].isna().any():
        raise ValueError(
            f"Column '{TEXT_COLUMN}' contains missing values."
        )

    if (df[TEXT_COLUMN].astype(str).str.strip() == "").any():
        raise ValueError(
            f"Column '{TEXT_COLUMN}' contains empty text."
        )

    if len(QUERIES) != 6:
        raise ValueError(
            f"Expected exactly 6 fixed queries, "
            f"but found {len(QUERIES)}."
        )

    if TOP_K != 5:
        raise ValueError(
            f"Expected TOP_K = 5, but found TOP_K = {TOP_K}."
        )

    return df

def build_vectorizer():
    return TfidfVectorizer(**TFIDF_CONFIG)

def retrieve_top_k(
    df,
    vectorizer,
    tfidf_matrix,
    query,
    query_id,
    top_k=TOP_K,
):
    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix,
    ).ravel()

    ranking = sorted(
        range(len(df)),
        key=lambda i: (
            -similarity_scores[i],
            df.iloc[i]["source_row_id"],
        ),
    )[:top_k]

    rows = []

    for rank, index in enumerate(ranking, start=1):
        movie = df.iloc[index]

        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "rank": rank,
                "source_row_id": movie["source_row_id"],
                "title": movie["title"],
                "genre": movie["genre"],
                "rating": movie["rating"],
                "similarity_score": float(
                    similarity_scores[index]
                ),
            }
        )

    return rows

def run_tfidf_retrieval(
    data_path=DATA_PATH,
    results_path=RESULTS_PATH,
):
    df = load_and_validate_corpus(data_path)

    vectorizer = build_vectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        df[TEXT_COLUMN].astype(str)
    )

    all_results = []

    for query_id, query in enumerate(
        QUERIES,
        start=1,
    ):
        query_results = retrieve_top_k(
            df=df,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            query=query,
            query_id=query_id,
            top_k=TOP_K,
        )

        all_results.extend(query_results)

    results_df = pd.DataFrame(
        all_results,
        columns=[
            "query_id",
            "query",
            "rank",
            "source_row_id",
            "title",
            "genre",
            "rating",
            "similarity_score",
        ],
    )

    results_path = Path(results_path)
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        results_df,
        vectorizer,
        tfidf_matrix,
    )

if __name__ == "__main__":
    results_df, vectorizer, tfidf_matrix = (
        run_tfidf_retrieval()
    )

    print("TF-IDF retrieval completed successfully.")
    print(f"Corpus records: {len(tfidf_matrix)}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Queries processed: {len(QUERIES)}")
    print(f"Top-K: {TOP_K}")
    print(f"Results generated: {len(results_df)}")
    print(f"Results file: {RESULTS_PATH}")