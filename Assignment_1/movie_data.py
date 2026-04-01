from __future__ import annotations

from pathlib import Path

import pandas as pd
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_MOVIE_CACHE_PATH = DATA_DIR / "processed_movies.pkl"


def load_data(data_dir=DATA_DIR):
    movies = pd.read_csv(Path(data_dir) / "movies.csv")
    ratings = pd.read_csv(Path(data_dir) / "ratings.csv")
    tags = pd.read_csv(Path(data_dir) / "tags.csv")
    return movies, ratings, tags


# Keep one row per movie and normalize the text fields used later.
def clean_movies(movies):
    movies = movies.drop_duplicates(subset=["movieId"]).dropna(subset=["movieId", "title"])
    movies["title"] = movies["title"].astype(str).str.strip()
    movies["genres"] = movies["genres"].fillna("").astype(str).str.strip()
    return movies

def clean_ratings(ratings):
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).drop_duplicates()
    return ratings

def clean_tags(tags):
    tags = tags.dropna(subset=["userId", "movieId", "tag"]).copy()
    tags["tag"] = (
        tags["tag"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    tags = tags[~tags["tag"].isin(["", "nan", "none", "null"])]
    tags = tags.drop_duplicates(subset=["userId", "movieId", "tag"])
    return tags

def clean_data(movies, ratings, tags):
    return clean_movies(movies), clean_ratings(ratings), clean_tags(tags)


# Makes the tag aggregation step more readable.
def join_unique_tags(values):
    return " ".join(pd.unique(values))


def build_movie_table_from_tables(ratings, movies, tags):
    rating_summary = (
        ratings.groupby("movieId", as_index=False)
        .agg(
            mean_rating=("rating", "mean"),
            rating_count=("rating", "count"),
        )
    )

    tag_summary = (
        tags.groupby("movieId", sort=False)["tag"]
        .apply(join_unique_tags)
        .reset_index(name="tag_text")
    )

    movie_table = (
        movies[["movieId", "title", "genres"]]
        .merge(rating_summary, on="movieId", how="left")
        .merge(tag_summary, on="movieId", how="left")
    )

    movie_table["mean_rating"] = movie_table["mean_rating"].fillna(0.0)
    movie_table["rating_count"] = movie_table["rating_count"].fillna(0).astype(int)
    movie_table["tag_text"] = movie_table["tag_text"].fillna("")
    movie_table["has_rating_and_tag"] = (movie_table["rating_count"] > 0) & (movie_table["tag_text"] != "")
    movie_table["clean_title"] = movie_table["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
    movie_table["genres_text"] = movie_table["genres"].str.replace("|", " ", regex=False)
    return movie_table


# Restrict the model data to movies that have both a rating signal and a tag signal.
def apply_movie_filters(
    movie_table,
    require_rating_and_tag=True,
):
    if not require_rating_and_tag:
        return movie_table.copy()

    return movie_table.loc[movie_table["has_rating_and_tag"]].copy()


# Runs the preprocessing functions end-to-end.
# The caching makes repeated runs faster.
def build_movie_table(
    data_dir=DATA_DIR,
    require_rating_and_tag=True,
    cache_path=DEFAULT_MOVIE_CACHE_PATH,
    use_cache=True,
    refresh_cache=False,
):
    cache_file = Path(cache_path)

    if use_cache and cache_file.exists() and not refresh_cache:
        movie_table = pd.read_pickle(cache_file)
        if "year_numeric" in movie_table.columns:
            movie_table = movie_table.drop(columns=["year_numeric"])
        if "year" in movie_table.columns:
            movie_table = movie_table.drop(columns=["year"])
        return movie_table

    movies, ratings, tags = load_data(data_dir)
    movies, ratings, tags = clean_data(movies, ratings, tags)
    movie_table = build_movie_table_from_tables(ratings, movies, tags)
    movie_table = apply_movie_filters(
        movie_table,
        require_rating_and_tag=require_rating_and_tag,
    )

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        movie_table.to_pickle(cache_file)

    return movie_table
