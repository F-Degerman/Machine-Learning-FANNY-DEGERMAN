import argparse

from config import RecommendationConfig
from recommender import build_recommender


# Needed for the command-line version of the project.
def parse_args():
    parser = argparse.ArgumentParser(description="Movie recommender for Assignment 1.")
    parser.add_argument("movie_title", nargs="?", help="Movie title to use as input.")
    parser.add_argument("--refresh-cache", action="store_true", help="Rebuild the cached movie table.")
    return parser.parse_args()

# Printing is necessary for the terminal interface in main.py.
def print_recommendations(recommendations):
    if recommendations.empty:
        print("No recommendations found.")
        return

    display_table = recommendations[["title", "mean_rating"]].copy()
    display_table = display_table.rename(columns={"mean_rating": "rating"})
    display_table["rating"] = display_table["rating"].map(lambda value: f"{value:.2f}")

    print("Recommendations")
    print(display_table.to_string(index=False))


def main():
    args = parse_args()
    model = build_recommender(
        config=RecommendationConfig(),
        use_cache=True,
        refresh_cache=args.refresh_cache,
    )
    movie_title = args.movie_title

    while True:
        movie_title = movie_title or input("Enter a movie title: ").strip()
        try:
            recommendations = model.recommend(movie_title)
            print_recommendations(recommendations)
            break
        except ValueError as error:
            print(error)
            print("Try again.")
            if args.movie_title:
                break
            movie_title = None


if __name__ == "__main__":
    main()
