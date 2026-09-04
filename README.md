# 🎬 Movie Recommendation System

A content-based movie recommendation engine built with **Python**, **Pandas**, and **Scikit-learn**. It suggests movies tailored to a user's taste by analyzing metadata (genres, keywords, cast, director, and plot overview) and ranking similarity with **cosine similarity** over **TF-IDF** vectors.

**Live demo:** _add your Streamlit Cloud URL here after deploying_

## Features

- Data preprocessing & feature extraction on movie metadata using Pandas
- TF-IDF vectorization of a weighted "content soup" (genres, keywords, cast, director, overview)
- Cosine similarity ranking to identify and recommend similar movies
- Interactive Streamlit web app
- CLI demo script for quick terminal testing

## Project Structure

```
movie-recommendation-system/
├── app.py                 # Streamlit web app (the "live" demo)
├── demo.py                # Command-line demo
├── requirements.txt
├── data/
│   └── movies.csv         # Movie metadata dataset
├── src/
│   └── recommender.py     # Core recommendation engine
└── README.md
```

## How It Works

1. **Preprocessing** — genres, keywords, cast, director, and overview columns are cleaned and combined into a single weighted text field per movie (genres/keywords/director are repeated to give them more influence than the overview).
2. **Feature Extraction** — the combined text is vectorized using `TfidfVectorizer` from scikit-learn.
3. **Similarity Ranking** — `cosine_similarity` computes pairwise similarity scores between all movies; the top-N most similar movies (excluding the input) are returned.

## Getting Started

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
```

### 2. Run the web app

```bash
streamlit run app.py
```

### 3. Or run the CLI demo

```bash
python demo.py "Inception" --top 5
```

## Example Output

```
Because you liked 'Inception', you might enjoy:

          title                  genres          director  similarity
     The Matrix           Action Sci-Fi    Lana Wachowski       0.261
The Dark Knight   Action Crime Thriller Christopher Nolan       0.193
  Black Panther Action Adventure Sci-Fi      Ryan Coogler       0.173
   Interstellar  Adventure Drama Sci-Fi Christopher Nolan       0.165
   The Prestige    Drama Mystery Sci-Fi Christopher Nolan       0.164
```

## Tech Stack

- Python
- Pandas
- Scikit-learn (TF-IDF, cosine similarity)
- Streamlit (web UI)

## Author

Yash Agarwal
