from pathlib import Path
import math
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "malayalam_df_clean.csv"
TFIDF_PATH = BASE_DIR / "results" / "tfidf" / "tfidf_results.csv"
TRANSFORMER_PATH = BASE_DIR / "results" / "transformer" / "transformer_results.csv"
OUTPUT_DIR = BASE_DIR / "results" / "evaluation"
EVALUATION_PATH = OUTPUT_DIR / "evaluation_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "comparison_summary.csv"


def validate_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required file is empty: {path}")


def load_data():
    validate_file(DATA_PATH)
    validate_file(TFIDF_PATH)
    validate_file(TRANSFORMER_PATH)

    corpus = pd.read_csv(DATA_PATH)
    tfidf = pd.read_csv(TFIDF_PATH)
    transformer = pd.read_csv(TRANSFORMER_PATH)

    required_corpus_columns = {
        "source_row_id",
        "title",
        "clean_synopsis"
    }

    required_result_columns = {
        "query_id",
        "query",
        "rank",
        "source_row_id",
        "title",
        "similarity_score"
    }

    if not required_corpus_columns.issubset(corpus.columns):
        missing = required_corpus_columns - set(corpus.columns)
        raise ValueError(f"Missing corpus columns: {sorted(missing)}")

    if not required_result_columns.issubset(tfidf.columns):
        missing = required_result_columns - set(tfidf.columns)
        raise ValueError(f"Missing TF-IDF columns: {sorted(missing)}")

    if not required_result_columns.issubset(transformer.columns):
        missing = required_result_columns - set(transformer.columns)
        raise ValueError(f"Missing Transformer columns: {sorted(missing)}")

    if len(corpus) != 94:
        raise ValueError(f"Expected 94 corpus records, found {len(corpus)}")

    if len(tfidf) != 30:
        raise ValueError(f"Expected 30 TF-IDF results, found {len(tfidf)}")

    if len(transformer) != 30:
        raise ValueError(f"Expected 30 Transformer results, found {len(transformer)}")

    return corpus, tfidf, transformer


def create_evaluation_file(corpus, tfidf, transformer):
    if EVALUATION_PATH.exists() and EVALUATION_PATH.stat().st_size > 0:
        return False

    records = []

    systems = [
        ("TF-IDF", tfidf),
        ("Transformer", transformer)
    ]

    for system_name, results in systems:
        for _, row in results.sort_values(
            ["query_id", "rank"]
        ).iterrows():
            source_row_id = int(row["source_row_id"])

            corpus_match = corpus[
                corpus["source_row_id"] == source_row_id
            ]

            if corpus_match.empty:
                raise ValueError(
                    f"source_row_id {source_row_id} not found in corpus"
                )

            synopsis = corpus_match.iloc[0]["clean_synopsis"]

            records.append(
                {
                    "system": system_name,
                    "query_id": int(row["query_id"]),
                    "query": row["query"],
                    "rank": int(row["rank"]),
                    "source_row_id": source_row_id,
                    "title": row["title"],
                    "similarity_score": float(row["similarity_score"]),
                    "synopsis": synopsis,
                    "relevance": ""
                }
            )

    evaluation_df = pd.DataFrame(records)

    expected_columns = [
        "system",
        "query_id",
        "query",
        "rank",
        "source_row_id",
        "title",
        "similarity_score",
        "synopsis",
        "relevance"
    ]

    evaluation_df = evaluation_df[expected_columns]

    if len(evaluation_df) != 60:
        raise ValueError(
            f"Expected 60 evaluation records, found {len(evaluation_df)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_df.to_csv(
        EVALUATION_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    return True


def load_evaluation_file():
    validate_file(EVALUATION_PATH)

    evaluation_df = pd.read_csv(
        EVALUATION_PATH,
        keep_default_na=False
    )

    required_columns = {
        "system",
        "query_id",
        "query",
        "rank",
        "source_row_id",
        "title",
        "similarity_score",
        "synopsis",
        "relevance"
    }

    if not required_columns.issubset(evaluation_df.columns):
        missing = required_columns - set(evaluation_df.columns)
        raise ValueError(
            f"Missing evaluation columns: {sorted(missing)}"
        )

    if len(evaluation_df) != 60:
        raise ValueError(
            f"Expected 60 evaluation records, found {len(evaluation_df)}"
        )

    return evaluation_df


def validate_relevance(evaluation_df):
    relevance_values = []

    for value in evaluation_df["relevance"]:
        if str(value).strip() == "":
            raise ValueError(
                "Some relevance values are still empty. "
                "Enter 0, 1, or 2 for every row."
            )

        try:
            numeric_value = int(float(value))
        except ValueError:
            raise ValueError(
                f"Invalid relevance value: {value}. "
                "Only 0, 1, or 2 are allowed."
            )

        if numeric_value not in {0, 1, 2}:
            raise ValueError(
                f"Invalid relevance value: {numeric_value}. "
                "Only 0, 1, or 2 are allowed."
            )

        relevance_values.append(numeric_value)

    evaluation_df["relevance"] = relevance_values

    return evaluation_df


def precision_at_5(relevances):
    relevant = sum(1 for value in relevances if value > 0)
    return relevant / 5


def dcg_at_5(relevances):
    score = 0.0

    for position, relevance in enumerate(relevances, start=1):
        score += (2 ** relevance - 1) / math.log2(position + 1)

    return score


def ndcg_at_5(relevances):
    actual_dcg = dcg_at_5(relevances)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_5(ideal_relevances)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def calculate_metrics(evaluation_df):
    records = []

    for system_name in ["TF-IDF", "Transformer"]:
        system_df = evaluation_df[
            evaluation_df["system"] == system_name
        ]

        for query_id in sorted(system_df["query_id"].unique()):
            query_df = system_df[
                system_df["query_id"] == query_id
            ].sort_values("rank")

            relevances = query_df["relevance"].tolist()

            if len(relevances) != 5:
                raise ValueError(
                    f"{system_name}, query {query_id} does not have 5 results"
                )

            records.append(
                {
                    "system": system_name,
                    "query_id": int(query_id),
                    "query": query_df.iloc[0]["query"],
                    "top1_relevance": relevances[0],
                    "precision_at_5": precision_at_5(relevances),
                    "dcg_at_5": dcg_at_5(relevances),
                    "ndcg_at_5": ndcg_at_5(relevances)
                }
            )

    query_metrics = pd.DataFrame(records)

    summary_records = []

    for system_name in ["TF-IDF", "Transformer"]:
        system_metrics = query_metrics[
            query_metrics["system"] == system_name
        ]

        summary_records.append(
            {
                "system": system_name,
                "queries_evaluated": len(system_metrics),
                "mean_top1_relevance": system_metrics[
                    "top1_relevance"
                ].mean(),
                "mean_precision_at_5": system_metrics[
                    "precision_at_5"
                ].mean(),
                "mean_dcg_at_5": system_metrics[
                    "dcg_at_5"
                ].mean(),
                "mean_ndcg_at_5": system_metrics[
                    "ndcg_at_5"
                ].mean()
            }
        )

    summary_df = pd.DataFrame(summary_records)

    return query_metrics, summary_df


def main():
    corpus, tfidf, transformer = load_data()

    created = create_evaluation_file(
        corpus,
        tfidf,
        transformer
    )

    if created:
        print("Evaluation file created.")
        print(f"File: {EVALUATION_PATH}")
        print("Rows: 60")
        print()
        print("Enter relevance values in the relevance column.")
        print("Use 2 for highly relevant.")
        print("Use 1 for partially relevant.")
        print("Use 0 for not relevant.")
        print()
        print("After completing all 60 judgments, run this program again.")
        return

    evaluation_df = load_evaluation_file()
    evaluation_df = validate_relevance(evaluation_df)

    query_metrics, summary_df = calculate_metrics(
        evaluation_df
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    evaluation_df.to_csv(
        EVALUATION_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    query_metrics.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Stage 4 evaluation completed.")
    print(f"Evaluation rows: {len(evaluation_df)}")
    print(f"Query metric rows: {len(query_metrics)}")
    print(f"Evaluation file: {EVALUATION_PATH}")
    print(f"Summary file: {SUMMARY_PATH}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()