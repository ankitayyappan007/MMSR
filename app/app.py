import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = BASE_DIR / "results" / "transformer" / "movie_embeddings.npy"
CORPUS_PATH = BASE_DIR / "data" / "processed" / "malayalam_df_clean.csv"
MODEL_NAME = "intfloat/multilingual-e5-base"
TOP_K = 5
MAX_SEQUENCE_LENGTH = 512

st.set_page_config(
    page_title="MMSR | Malayalam Cinema Retrieval",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(120, 30, 50, 0.18), transparent 35%),
            radial-gradient(circle at top right, rgba(30, 60, 100, 0.16), transparent 35%),
            #0b0d12;
        color: #f4f1ea;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    .hero {
        padding: 2.5rem 2.8rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(28, 30, 40, 0.96), rgba(15, 17, 24, 0.96));
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
    }

    .eyebrow {
        color: #c9a86a;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        color: #f7f3eb;
        font-size: 2.45rem;
        font-weight: 700;
        line-height: 1.15;
        margin: 0;
    }

    .hero-subtitle {
        color: #aeb2bd;
        font-size: 1rem;
        line-height: 1.7;
        margin-top: 1rem;
        max-width: 850px;
    }

    

    .section-label {
        color: #c9a86a;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .result-card {
        padding: 1.4rem 1.5rem;
        margin: 0.9rem 0;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background: linear-gradient(145deg, #171a22, #11131a);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
    }

    .result-card-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1.5rem;
    }

    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 82px;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: rgba(201, 168, 106, 0.12);
        color: #d7bd86;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .rank-one {
        border-color: rgba(201, 168, 106, 0.48);
        box-shadow: 0 14px 40px rgba(201, 168, 106, 0.10);
    }

    .rank-one .rank-badge {
        background: #c9a86a;
        color: #17130c;
    }

    .movie-title {
        color: #f5f1e8;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.7rem;
    }

    .movie-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-bottom: 1rem;
    }

    .meta-pill {
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        color: #bfc3cc;
        font-size: 0.78rem;
    }

    .score {
        color: #e4c987;
        font-size: 1.25rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .score-label {
        color: #858a96;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        display: block;
        text-align: right;
        margin-bottom: 0.15rem;
    }

    .synopsis-label {
        color: #8f94a0;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .synopsis {
        color: #c4c7cf;
        font-size: 0.93rem;
        line-height: 1.7;
        margin: 0;
    }

    .footer-note {
        text-align: center;
        color: #6f7480;
        font-size: 0.75rem;
        margin-top: 2.5rem;
    }

    textarea {
        background-color: #11131a !important;
        color: #f5f1e8 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
    }

    textarea::placeholder {
        color: #777c88 !important;
    }

    div.stButton > button {
        width: 100%;
        min-height: 3rem;
        border: 1px solid #c9a86a;
        border-radius: 12px;
        background: linear-gradient(135deg, #c9a86a, #a9874d);
        color: #17130c;
        font-weight: 700;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #e0c88f;
        background: linear-gradient(135deg, #e0c88f, #b99559);
        color: #17130c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Semantic Movie Script Retrieval for Malayalam Cinema")
st.write(
    "Describe a story, theme, or plot idea and retrieve the five most semantically relevant Malayalam movies from the evaluated cinema corpus."
)

@st.cache_resource
def load_model():
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    return model

@st.cache_data
def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)
    corpus = pd.read_csv(CORPUS_PATH)

    if len(embeddings) != len(corpus):
        raise ValueError(
            "The saved embeddings and processed corpus contain different numbers of records."
        )

    if embeddings.ndim != 2:
        raise ValueError("The saved movie embeddings must be a two-dimensional array.")

    return embeddings, corpus

model = load_model()
embeddings, corpus = load_data()



st.markdown(
    '<div class="section-label">Story / Plot Description</div>',
    unsafe_allow_html=True
)

query = st.text_area(
    "Story / Plot Description",
    placeholder="Example: A story about a family facing conflicts and trying to stay together.",
    height=150,
    label_visibility="collapsed"
)

search_clicked = st.button("Search Movies", type="primary")



if search_clicked:
    query = query.strip()

    if not query:
        st.warning("Please enter a story or plot description.")
    else:
        query_embedding = model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        scores = embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:TOP_K]

        st.markdown(
            '<div class="section-label">Top 5 Results</div>',
            unsafe_allow_html=True
        )

        for rank, index in enumerate(top_indices, start=1):
            row = corpus.iloc[index]

            title = row["title"] if "title" in corpus.columns else f"Movie {index + 1}"
            genre = row["genre"] if "genre" in corpus.columns else "Not available"
            rating = row["rating"] if "rating" in corpus.columns else "Not available"
            synopsis = row["synopsis"] if "synopsis" in corpus.columns else "Synopsis not available"

            if pd.isna(genre) or str(genre).strip() == "":
                genre = "Not available"

            if pd.isna(rating) or str(rating).strip() == "":
                rating = "Not available"

            if pd.isna(synopsis) or str(synopsis).strip() == "":
                synopsis = "Synopsis not available"

            card_class = "result-card rank-one" if rank == 1 else "result-card"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div class="result-card-top">
                        <div>
                            <div class="rank-badge">Rank {rank}</div>
                            <div class="movie-title">{title}</div>
                        </div>
                        <div>
                            <span class="score-label">Similarity</span>
                            <div class="score">{scores[index]:.4f}</div>
                        </div>
                    </div>
                    <div class="movie-meta">
                        <span class="meta-pill">Genre: {genre}</span>
                        <span class="meta-pill">Rating: {rating}</span>
                    </div>
                    <div class="synopsis-label">Synopsis</div>
                    <p class="synopsis">{synopsis}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown(
    '<div class="footer-note">MMSR · Transformer-based semantic retrieval · Top-K = 5</div>',
    unsafe_allow_html=True
)