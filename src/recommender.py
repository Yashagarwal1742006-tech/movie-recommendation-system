"""
Content-Based Movie Recommendation Engine
-------------------------------------------
Builds a "content soup" from each movie's genres, keywords, cast,
director and overview, vectorizes it with TF-IDF, and ranks similar
movies using cosine similarity.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, data_path: str = "data/movies.csv"):
        self.data_path = data_path
        self.df = None
        self.similarity_matrix = None
        self.indices = None
        self._load_and_prepare()

    # ------------------------------------------------------------------ #
    # Data preprocessing & feature extraction
    # ------------------------------------------------------------------ #
    def _load_and_prepare(self):
        df = pd.read_csv(self.data_path)

        # Basic cleaning
        text_cols = ["genres", "keywords", "overview", "cast", "director"]
        for col in text_cols:
            df[col] = df[col].fillna("").astype(str)

        # Give cast/director/genres extra weight by repeating them,
        # a common content-based-filtering trick so metadata dominates
        # over generic overview words.
        df["soup"] = (
            (df["genres"] + " ") * 3
            + (df["keywords"] + " ") * 2
            + (df["cast"].str.replace(" ", "", regex=False) + " ")
            + (df["director"].str.replace(" ", "", regex=False) + " ") * 2
            + df["overview"]
        ).str.lower()

        df = df.reset_index(drop=True)
        self.df = df
        self.indices = pd.Series(df.index, index=df["title"].str.lower())

        # TF-IDF vectorization + cosine similarity
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df["soup"])
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def recommend(self, title: str, top_n: int = 5) -> pd.DataFrame:
        """Return the top_n movies most similar to `title`."""
        key = title.lower().strip()
        if key not in self.indices:
            matches = self.df[self.df["title"].str.lower().str.contains(key)]
            if matches.empty:
                raise ValueError(f"Movie '{title}' not found in the dataset.")
            idx = matches.index[0]
        else:
            idx = self.indices[key]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]

        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        movie_indices = [i for i, _ in scores]
        result = self.df.iloc[movie_indices][["title", "genres", "director"]].copy()
        result["similarity"] = [round(s, 3) for _, s in scores]
        return result.reset_index(drop=True)

    def titles(self):
        return sorted(self.df["title"].tolist())


if __name__ == "__main__":
    engine = MovieRecommender()
    sample = "Inception"
    print(f"Movies similar to '{sample}':\n")
    print(engine.recommend(sample, top_n=5).to_string(index=False))
