from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "malayalam_df_clean.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "transformer"
EMBEDDINGS_PATH = RESULTS_DIR / "movie_embeddings.npy"
RESULTS_PATH = RESULTS_DIR / "transformer_results.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_config import QUERIES, TOP_K

TEXT_COLUMN = "clean_synopsis"
EXPECTED_ROW_COUNT = 94
EXPECTED_EMBEDDING_DIMENSION = 768
MAX_SEQUENCE_LENGTH = 512
MODEL_NAME = "intfloat/multilingual-e5-base"
DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "

REQUIRED_COLUMNS = {
    "source_row_id",
    "title",
    "synopsis",
    "genre",
    "rating",
    "movie_language",
    "title_confidence",
    TEXT_COLUMN,
}


def load_and_validate_corpus(data_path=DATA_PATH):
    data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if len(df) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} records, but found {len(df)}."
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
            f"Expected 6 queries, but found {len(QUERIES)}."
        )

    if TOP_K != 5:
        raise ValueError(
            f"Expected TOP_K to be 5, but found {TOP_K}."
        )

    return df


def load_model():
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    return model


def encode_documents(model, df):
    documents = [
        DOCUMENT_PREFIX + text
        for text in df[TEXT_COLUMN].astype(str)
    ]

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (
        EXPECTED_ROW_COUNT,
        EXPECTED_EMBEDDING_DIMENSION
    ):
        raise ValueError(
            "Unexpected embedding shape: "
            + str(embeddings.shape)
        )

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(
            "Document embeddings are not L2-normalized."
        )

    return embeddings


def encode_queries(model):
    query_inputs = [
        QUERY_PREFIX + query
        for query in QUERIES
    ]

    query_embeddings = model.encode(
        query_inputs,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32
    )

    if query_embeddings.shape != (
        len(QUERIES),
        EXPECTED_EMBEDDING_DIMENSION
    ):
        raise ValueError(
            "Unexpected query embedding shape: "
            + str(query_embeddings.shape)
        )

    norms = np.linalg.norm(query_embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(
            "Query embeddings are not L2-normalized."
        )

    return query_embeddings


def retrieve_top_k(df, query_embeddings, document_embeddings):
    all_results = []

    for query_id, query_embedding in enumerate(
        query_embeddings,
        start=1
    ):
        similarity_scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            document_embeddings
        ).ravel()

        ranking = sorted(
            range(len(df)),
            key=lambda i: (
                -similarity_scores[i],
                df.iloc[i]["source_row_id"]
            )
        )[:TOP_K]

        for rank, index in enumerate(ranking, start=1):
            movie = df.iloc[index]

            all_results.append({
                "query_id": query_id,
                "query": QUERIES[query_id - 1],
                "rank": rank,
                "source_row_id": movie["source_row_id"],
                "title": movie["title"],
                "genre": movie["genre"],
                "rating": movie["rating"],
                "similarity_score": float(
                    similarity_scores[index]
                )
            })

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
            "similarity_score"
        ]
    )

    expected_results = len(QUERIES) * TOP_K

    if len(results_df) != expected_results:
        raise ValueError(
            f"Expected {expected_results} results, "
            f"but generated {len(results_df)}."
        )

    return results_df


def save_embeddings(embeddings, embeddings_path=EMBEDDINGS_PATH):
    embeddings_path = Path(embeddings_path)
    embeddings_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(embeddings_path, embeddings)


def save_results(results_df, results_path=RESULTS_PATH):
    results_path = Path(results_path)
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig"
    )


def run_transformer_retrieval():
    df = load_and_validate_corpus()

    print(f"Corpus records: {len(df)}")

    model = load_model()

    print(f"Model loaded: {MODEL_NAME}")
    print(f"Maximum sequence length: {model.max_seq_length}")

    document_embeddings = encode_documents(
        model,
        df
    )

    print(
        "Document embeddings shape:",
        document_embeddings.shape
    )

    save_embeddings(
        document_embeddings,
        EMBEDDINGS_PATH
    )

    query_embeddings = encode_queries(model)

    print(
        "Query embeddings shape:",
        query_embeddings.shape
    )

    results_df = retrieve_top_k(
        df,
        query_embeddings,
        document_embeddings
    )

    save_results(
        results_df,
        RESULTS_PATH
    )

    print("Transformer retrieval completed successfully.")
    print(f"Queries processed: {len(QUERIES)}")
    print(f"Top-K: {TOP_K}")
    print(f"Results generated: {len(results_df)}")
    print(f"Embeddings saved to: {EMBEDDINGS_PATH}")
    print(f"Results saved to: {RESULTS_PATH}")

    return results_df, document_embeddings, query_embeddings


if __name__ == "__main__":
    run_transformer_retrieval()