import streamlit as st
from src.recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Content-Based Movie Recommendation System")
st.caption("Python · Pandas · Scikit-learn · TF-IDF · Cosine Similarity")

st.write(
    "Pick a movie you like and the engine will rank the most similar "
    "titles using a content-based model built on genres, keywords, cast, "
    "director and plot overview."
)


@st.cache_resource
def load_engine():
    return MovieRecommender(data_path="data/movies.csv")


engine = load_engine()

movie = st.selectbox("Choose a movie", engine.titles())
top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

if st.button("Get Recommendations", type="primary"):
    results = engine.recommend(movie, top_n=top_n)
    st.subheader(f"Movies similar to '{movie}'")
    for _, row in results.iterrows():
        st.markdown(
            f"**{row['title']}** — *{row['genres']}* "
            f"(dir. {row['director']}) · similarity: `{row['similarity']}`"
        )

st.divider()
with st.expander("How it works"):
    st.write(
        "Each movie's genres, keywords, cast, director and overview are "
        "combined into a single text 'soup' (with genres/keywords/director "
        "weighted more heavily), vectorized with TF-IDF, and compared "
        "pairwise using cosine similarity. The highest-scoring movies "
        "(excluding the selected one) are returned as recommendations."
    )
