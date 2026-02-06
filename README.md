# Book-Recommendation-Model

A Streamlit book recommender with a **hybrid strategy**:

- Collaborative filtering signal from the book-user pivot matrix.
- Genre-aware re-ranking based on user-selected genres.
- Cleaner, modern UI for easier exploration.

## Run locally

```bash
pip install -r requirement.txt
streamlit run app.py
```

## How recommendations are generated

1. Pick a seed book from the catalog.
2. Compute cosine similarity between the selected book vector and all other books.
3. Infer lightweight genre tags from title keywords.
4. Boost books that match selected genres.
5. Return top-ranked recommendations.

> Note: Genre inference is currently keyword-based from book titles. For higher-quality genre recommendations, include explicit genre metadata in the training dataset.
