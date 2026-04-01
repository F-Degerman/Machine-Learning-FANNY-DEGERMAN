# Assignment 1

This folder contains a movie recommendation system built on MovieLens-style data.

The recommender is content-based and works by:

- building one movie-level table from movie, rating, and tag data
- retrieving similar movies with TF-IDF features and `NearestNeighbors`
- reranking candidates with similarity and rating information
- using `KMeans` to diversify the final recommendation list

## Files

- [main.py]
  Command-line entry point.
- [recommender.py]
  Main recommendation flow, title matching, candidate preparation, reranking, and diversification.
- [modeling.py]
  Vectorizers, feature transformers, and sklearn pipelines.
- [movie_data.py]
  Data loading, cleaning, aggregation, filtering, and caching.
- [config.py]
  Central configuration and hyperparameters.
- [EDA.ipynb]
  Exploratory data analysis notebook.
- [report.ipynb]
  Final report notebook.

## Expected Data

The code expects the following files in the repository `data` folder:

- `movies.csv`
- `ratings.csv`
- `tags.csv`

## Model Summary

The current flow is:

1. Load and clean the raw movie, rating, and tag tables.
2. Build a movie-level table with `genres_text`, `tag_text`, `mean_rating`, `rating_count`, and `clean_title`.
3. Filter the modeling table to movies that have both ratings and tags.
4. Retrieve a larger candidate pool with TF-IDF on genres and tags plus `NearestNeighbors`.
5. Apply a mild candidate-quality filter.
6. Rerank candidates with a combined score based on similarity and normalized mean rating.
7. Cluster the candidate pool with `KMeans`.
8. Select one strong movie per cluster first, then fill remaining slots from the best leftovers if needed.

