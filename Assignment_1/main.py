import argparse

from recommender import build_recommender
from training_model import RecommendationConfig


# This parser is only needed for the command-line version of the project.
def parse_args():
    """Read the CLI input."""
    parser = argparse.ArgumentParser(description="Movie recommender for Assignment 1.")
    parser.add_argument("movie_title", help="Movie title to use as input.")
    parser.add_argument("--refresh-cache", action="store_true", help="Rebuild the cached movie table.")
    return parser.parse_args()


# This helper is not strictly necessary, but it keeps the main function short and
# makes it clear where the application creates the fitted recommender object.
def build_application_model(refresh_cache=False):
    """Build the recommender once before any user query is handled."""
    return build_recommender(
        config=RecommendationConfig(),
        use_cache=True,
        refresh_cache=refresh_cache,
    )


# Printing is application logic only. It is not necessary for the recommender
# object, but it is necessary for the terminal interface in main.py.
def print_recommendations(recommendations):
    """Print the final recommendation list returned to the user."""
    if recommendations.empty:
        print("No recommendations found.")
        return

    print("Recommendations")
    print(recommendations[["title", "similarity_score", "mean_rating", "rating_count"]].to_string(index=False))


def main():
    args = parse_args()
    model = build_application_model(refresh_cache=args.refresh_cache)
    recommendations = model.recommend(args.movie_title)
    print_recommendations(recommendations)


if __name__ == "__main__":
    main()
