"""
Command-line demo for the Movie Recommendation Engine.

Usage:
    python demo.py "Inception"
    python demo.py "The Dark Knight" --top 8
"""

import argparse
from src.recommender import MovieRecommender


def main():
    parser = argparse.ArgumentParser(description="Get movie recommendations.")
    parser.add_argument("title", type=str, help="Movie title to base recommendations on")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations")
    args = parser.parse_args()

    engine = MovieRecommender(data_path="data/movies.csv")
    try:
        results = engine.recommend(args.title, top_n=args.top)
    except ValueError as e:
        print(e)
        print("\nAvailable titles:")
        for t in engine.titles():
            print(f"  - {t}")
        return

    print(f"\nBecause you liked '{args.title}', you might enjoy:\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
