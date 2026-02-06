from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import streamlit as st


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")


# -----------------------------
# Recommendation helpers
# -----------------------------
GENRE_KEYWORDS: dict[str, list[str]] = {
    "Fantasy": ["dragon", "magic", "wizard", "sorcer", "kingdom", "sword", "chronicles"],
    "Science Fiction": ["space", "robot", "future", "galaxy", "time", "planet", "alien"],
    "Mystery & Thriller": ["murder", "death", "secret", "case", "thriller", "detective", "crime"],
    "Romance": ["love", "heart", "kiss", "romance", "wedding", "bride"],
    "Historical": ["history", "war", "empire", "queen", "king", "century"],
    "Horror": ["ghost", "haunt", "dark", "night", "horror", "vampire"],
    "Young Adult": ["school", "teen", "young", "academy", "girl", "boy"],
    "Classics": ["pride", "prejudice", "odyssey", "mockingbird", "gatsby", "1984"],
}


def _resolve_data_path(file_name: str) -> Path:
    """Resolve file paths from artifacts/ first, then project root."""
    candidates = [Path("artifacts") / file_name, Path(file_name)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing required file: {file_name}")


@st.cache_resource
def load_model_resource() -> object:
    """Load optional model resource if available."""
    try:
        model_path = _resolve_data_path("model.pkl")
        with model_path.open("rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None


@st.cache_data
def load_data_assets() -> tuple[list[str], object, object | None]:
    """Load datasets safely and return book names, pivot matrix, and ratings metadata."""
    book_name_path = _resolve_data_path("book_name.pkl")
    book_pivot_path = _resolve_data_path("book_pivot.pkl")

    with book_name_path.open("rb") as file:
        raw_book_names = pickle.load(file)

    with book_pivot_path.open("rb") as file:
        book_pivot = pickle.load(file)

    try:
        final_rating_path = _resolve_data_path("final_rating.pkl")
        with final_rating_path.open("rb") as file:
            final_ratings = pickle.load(file)
    except FileNotFoundError:
        final_ratings = None

    # Handle pandas Index / numpy arrays / python lists safely.
    if hasattr(raw_book_names, "tolist"):
        book_names = [str(name) for name in raw_book_names.tolist()]
    else:
        book_names = [str(name) for name in list(raw_book_names)]

    return book_names, book_pivot, final_ratings


def infer_title_genres(title: str) -> set[str]:
    """Infer lightweight genre tags from title keywords."""
    lowered = title.lower()
    matched = {
        genre
        for genre, keywords in GENRE_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    }
    return matched or {"General"}


def cosine_similarities(seed_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarities for all rows against a seed vector."""
    seed_norm = np.linalg.norm(seed_vector) + 1e-12
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ seed_vector) / (matrix_norm * seed_norm)


def recommend_books(
    selected_book: str,
    selected_genres: list[str],
    book_pivot: object,
    top_n: int = 10,
) -> list[dict[str, object]]:
    """Generate top book recommendations with genre-aware re-ranking."""
    if selected_book not in set(book_pivot.index):
        return []

    seed_idx = int(np.where(book_pivot.index == selected_book)[0][0])
    matrix = np.asarray(book_pivot.values, dtype=np.float32)

    if matrix.size == 0:
        return []

    sims = cosine_similarities(matrix[seed_idx], matrix)
    ranked_indices = np.argsort(-sims)

    recommendations: list[dict[str, object]] = []
    selected_genres_set = set(selected_genres)

    for idx in ranked_indices:
        if idx == seed_idx:
            continue

        title = str(book_pivot.index[idx])
        inferred_genres = infer_title_genres(title)
        match_count = len(selected_genres_set.intersection(inferred_genres))

        score = float(sims[idx]) + (0.06 * match_count)
        recommendations.append(
            {
                "title": title,
                "genre": ", ".join(sorted(inferred_genres)),
                "score": max(0.0, min(score, 1.0)),
            }
        )

    return recommendations[:top_n]


# -----------------------------
# UI
# -----------------------------
with st.container():
    left_pad, center_col, right_pad = st.columns([1, 3, 1])
    with center_col:
        st.title("📚 Book Recommendation System")

st.divider()

try:
    _ = load_model_resource()  # Kept cached for compatibility with existing artifacts.
    book_names, book_pivot, _ = load_data_assets()
except Exception as exc:
    st.error(f"Unable to load data assets: {exc}")
    st.stop()

if not book_names:
    st.warning("No books found in the dataset.")
    st.stop()

with st.container():
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        selected_book = st.selectbox("Select a book", options=book_names, index=0)

    with input_col2:
        selected_genres = st.multiselect(
            "Select preferred genres",
            options=list(GENRE_KEYWORDS.keys()),
            default=["Fantasy", "Mystery & Thriller"],
        )

recommend_clicked = st.button("✨ Recommend Books", use_container_width=True, type="primary")

st.divider()
st.subheader("Recommendations")

if not recommend_clicked:
    st.info("No recommendations yet")
else:
    with st.spinner("Finding the best books for you..."):
        recommendations = recommend_books(selected_book, selected_genres, book_pivot)

    if not recommendations:
        st.info("No recommendations yet")
    else:
        for row_start in range(0, len(recommendations), 2):
            row_cols = st.columns(2)
            for col_idx in range(2):
                rec_idx = row_start + col_idx
                if rec_idx >= len(recommendations):
                    continue

                rec = recommendations[rec_idx]
                with row_cols[col_idx]:
                    with st.container(border=True):
                        st.markdown(f"**{rec['title']}**")
                        st.caption(f"Genre: {rec['genre']}")
                        st.caption("Recommendation score")
                        st.progress(float(rec["score"]))

st.divider()
st.caption("Built with Streamlit")
