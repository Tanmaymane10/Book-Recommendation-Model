from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import streamlit as st


st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
)


GENRE_KEYWORDS = {
    "Fantasy": ["dragon", "magic", "wizard", "sorcer", "kingdom", "sword", "chronicles"],
    "Science Fiction": ["space", "robot", "future", "galaxy", "time", "planet", "alien"],
    "Mystery & Thriller": ["murder", "death", "secret", "case", "thriller", "detective", "crime"],
    "Romance": ["love", "heart", "kiss", "romance", "wedding", "bride"],
    "Historical": ["history", "war", "empire", "queen", "king", "century"],
    "Horror": ["ghost", "haunt", "dark", "night", "horror", "vampire"],
    "Young Adult": ["school", "teen", "young", "academy", "girl", "boy"],
    "Classics": ["pride", "prejudice", "odyssey", "mockingbird", "gatsby", "1984"],
}


def _load_pickle(file_name: str):
    candidates = [Path("artifacts") / file_name, Path(file_name)]
    for path in candidates:
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Could not find {file_name} in artifacts/ or project root.")


@st.cache_resource
def load_assets():
    books_name = _load_pickle("book_name.pkl")
    book_pivot = _load_pickle("book_pivot.pkl")

    try:
        final_ratings = _load_pickle("final_rating.pkl")
    except FileNotFoundError:
        final_ratings = None

    # Optional saved KNN model (from notebook). The app can run without it.
    try:
        model = _load_pickle("model.pkl")
    except FileNotFoundError:
        model = None

    return books_name, book_pivot, final_ratings, model


def infer_title_genres(title: str) -> set[str]:
    lowered = title.lower()
    matched = {
        genre
        for genre, keywords in GENRE_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    }
    return matched or {"General"}


def cosine_similarities(book_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    book_norm = np.linalg.norm(book_vector) + 1e-12
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ book_vector) / (matrix_norm * book_norm)


def recommend_books(book_name: str, selected_genres: list[str], book_pivot, top_n: int = 10):
    idx = np.where(book_pivot.index == book_name)[0][0]
    matrix = np.asarray(book_pivot.values, dtype=np.float32)

    sims = cosine_similarities(matrix[idx], matrix)
    ranking = np.argsort(-sims)

    recommendations = []
    for r_idx in ranking:
        if r_idx == idx:
            continue

        title = str(book_pivot.index[r_idx])
        title_genres = infer_title_genres(title)
        genre_match_count = len(set(selected_genres).intersection(title_genres))
        score = float(sims[r_idx]) + (0.06 * genre_match_count)

        recommendations.append(
            {
                "title": title,
                "score": score,
                "genre_tags": sorted(title_genres),
                "genre_matches": genre_match_count,
            }
        )

    recommendations = sorted(recommendations, key=lambda row: row["score"], reverse=True)
    return recommendations[:top_n]


def poster_for_title(title: str, final_ratings):
    if final_ratings is None:
        return None

    result = final_ratings[final_ratings["Title"] == title]
    if result.empty:
        return None

    image_url = result.iloc[0].get("image_url")
    return image_url if isinstance(image_url, str) and image_url.startswith("http") else None


books_name, book_pivot, final_ratings, model = load_assets()

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(120deg, #f5f7fa 0%, #e8ecf7 100%);}
    .hero {
      background: linear-gradient(135deg, #1f3b73 0%, #4566a8 100%);
      padding: 1.2rem 1.4rem;
      border-radius: 14px;
      color: white;
      margin-bottom: 1rem;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    .book-card {
      background: white;
      border-radius: 12px;
      padding: 0.8rem;
      border: 1px solid #e7eaf3;
      margin-bottom: 0.6rem;
      min-height: 150px;
      box-shadow: 0 3px 10px rgba(50, 73, 114, 0.08);
    }
    .tag {
      display: inline-block;
      padding: 0.1rem 0.55rem;
      border-radius: 999px;
      background: #edf1ff;
      color: #3858a5;
      font-size: 0.75rem;
      margin-right: 0.25rem;
      margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h2 style="margin:0;">📚 Smarter Book Recommendations</h2>
        <p style="margin:0.45rem 0 0 0;">
            Pick a book you like, choose your preferred genres, and get hybrid recommendations
            (similar reading patterns + genre-aware re-ranking).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2, 1])

with left:
    selected_book = st.selectbox("Choose a book", books_name)

with right:
    selected_genres = st.multiselect(
        "Preferred genres",
        options=list(GENRE_KEYWORDS.keys()),
        default=["Fantasy", "Mystery & Thriller"],
    )

if st.button("✨ Recommend books", use_container_width=True):
    results = recommend_books(selected_book, selected_genres, book_pivot)

    st.subheader("Top picks for you")
    row_cols = st.columns(2)

    for i, rec in enumerate(results):
        with row_cols[i % 2]:
            st.markdown('<div class="book-card">', unsafe_allow_html=True)
            st.markdown(f"**{rec['title']}**")
            st.caption(f"Recommendation score: {rec['score']:.3f}")

            tags_html = "".join([f'<span class="tag">{tag}</span>' for tag in rec["genre_tags"]])
            st.markdown(tags_html, unsafe_allow_html=True)

            poster_url = poster_for_title(rec["title"], final_ratings)
            if poster_url:
                st.image(poster_url, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Genre inference currently uses title keywords; add explicit genre metadata in the dataset for best accuracy."
)
