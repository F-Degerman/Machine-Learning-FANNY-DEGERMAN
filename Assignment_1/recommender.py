from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator

from movie_data import (
    DEFAULT_MOVIE_CACHE_PATH,
    DATA_DIR,
    apply_movie_filters,
    build_movie_table_from_tables,
    clean_data,
    load_data,
)
from training_model import (
    RecommendationConfig,
    build_baseline_pipeline,
    build_improved_pipeline,
)

MATCH_RANK_COLUMNS = ["rating_count", "mean_rating"]
DEFAULT_RANK_COLUMNS = ["similarity_score", "mean_rating", "rating_count"]
DEFAULT_RANK_ASCENDING = [False, False, False]

""" Helpers"""

def rank_candidate_rows(candidates):
    return candidates.sort_values(
        by=DEFAULT_RANK_COLUMNS,
        ascending=DEFAULT_RANK_ASCENDING,
    ).reset_index(drop=True)


def select_best_match(matches):
    return matches.sort_values(
        by=MATCH_RANK_COLUMNS,
        ascending=[False, False],
    ).index[0]


def normalize_title(title):
    return str(title).strip().lower()


#  recommendation flow for one coherent model object 
# instead of many loose functions.
class MovieRecommender(BaseEstimator):

    def __init__(self, config=RecommendationConfig()):
        self.config = config

    def fit(self, movie_table, y=None):
        """Fit the baseline pipeline on the filtered movie table."""
        self.movie_table_ = movie_table.reset_index(drop=True).copy()
        self.movie_table_["normalized_title"] = self.movie_table_["clean_title"].map(normalize_title)

        self.baseline_pipeline_ = build_baseline_pipeline(self.config)
        self.baseline_pipeline_.fit(self.movie_table_)
        return self

    def _resolve_movie_index(self, movie_title):
        """Find the movie row that best matches the title the user typed."""
        normalized_query = normalize_title(movie_title)

        exact_matches = self.movie_table_[self.movie_table_["normalized_title"] == normalized_query]
        if not exact_matches.empty:
            return select_best_match(exact_matches)

        partial_matches = self.movie_table_[
            self.movie_table_["normalized_title"].str.contains(normalized_query, regex=False)
        ]
        if not partial_matches.empty:
            return select_best_match(partial_matches)

        raise ValueError(f"No movie title match found for '{movie_title}'.")

    # To keep the candidate-building loop shorter
    # and avoid repeating the same row-to-dict mapping logic inline.
    def _build_candidate_row(self, movie_index, distance):
        movie_row = self.movie_table_.iloc[movie_index]
        return {
            "movieId": movie_row["movieId"],
            "title": movie_row["clean_title"],
            "year": movie_row["year"],
            "genres_text": movie_row["genres_text"],
            "tag_text": movie_row["tag_text"],
            "year_numeric": movie_row["year_numeric"],
            "mean_rating": movie_row["mean_rating"],
            "rating_count": movie_row["rating_count"],
            "similarity_score": 1 - float(distance),
        }

    # Retriev candidate pool for improved model
    def get_candidate_movies(self, movie_title, candidate_pool_size=None):
        query_index = self._resolve_movie_index(movie_title)
        query_row = self.movie_table_.iloc[[query_index]]
        query_features = self.baseline_pipeline_.named_steps["features"].transform(query_row)

        requested_neighbors = candidate_pool_size or self.config.diversity.candidate_pool_size
        n_neighbors = min(requested_neighbors + 1, len(self.movie_table_))
        distances, indices = self.baseline_pipeline_.named_steps["retriever"].kneighbors(
            query_features,
            n_neighbors=n_neighbors,
        )

        rows = [
            self._build_candidate_row(index, distance)
            for distance, index in zip(distances[0], indices[0])
            if index != query_index
        ]

        candidates = pd.DataFrame(rows)
        if candidates.empty:
            return candidates

        return rank_candidate_rows(candidates)

    # keeps the KMeans method focused on clustering
    # rather than also holding the full round-robin selection logic.
    def _select_diverse_rows(self, grouped_candidates, cluster_priority, n_recommendations):
        selected_rows = []
        cluster_positions = {cluster_id: 0 for cluster_id in cluster_priority}

        while len(selected_rows) < n_recommendations:
            added_in_round = False

            for cluster_id in cluster_priority:
                cluster_rows = grouped_candidates[cluster_id]
                position = cluster_positions[cluster_id]

                if position >= len(cluster_rows):
                    continue

                selected_rows.append(cluster_rows.iloc[position].to_dict())
                cluster_positions[cluster_id] += 1
                added_in_round = True

                if len(selected_rows) == n_recommendations:
                    break

            if not added_in_round:
                break

        return pd.DataFrame(selected_rows)

    # This helper is optional, but it keeps the main diversification method focused
    # on flow rather than on default-value and boundary handling.
    def _resolve_diversity_settings(self, candidates, n_recommendations, n_candidate_clusters):
        config = self.config.diversity
        n_recommendations = n_recommendations or config.n_recommendations
        n_candidate_clusters = n_candidate_clusters or config.n_candidate_clusters

        if len(candidates) <= n_recommendations:
            return n_recommendations, None

        n_clusters = min(n_candidate_clusters, n_recommendations, len(candidates))
        if n_clusters < 2:
            return n_recommendations, None

        return n_recommendations, n_clusters

    # This helper is optional, but it keeps the actual KMeans application isolated
    # from the later ranking and round-robin selection logic.
    def _cluster_candidates(self, candidates, n_clusters):
        improved_pipeline = build_improved_pipeline(
            self.config,
            n_clusters=n_clusters,
        )
        improved_pipeline.fit(candidates)
        cluster_labels = improved_pipeline.named_steps["clusterer"].labels_

        clustered_candidates = candidates.copy()
        clustered_candidates["cluster"] = cluster_labels
        return clustered_candidates.sort_values(
            by=["cluster", *DEFAULT_RANK_COLUMNS],
            ascending=[True, *DEFAULT_RANK_ASCENDING],
        )

    # This helper is optional, but it keeps the prioritization rule explicit:
    # clusters are traversed starting from the one with the strongest top candidate.
    def _rank_clusters(self, clustered_candidates):
        grouped_candidates = {
            cluster_id: group.reset_index(drop=True)
            for cluster_id, group in clustered_candidates.groupby("cluster", sort=False)
        }
        cluster_priority = sorted(
            grouped_candidates,
            key=lambda cluster_id: grouped_candidates[cluster_id].iloc[0]["similarity_score"],
            reverse=True,
        )
        return grouped_candidates, cluster_priority

    def _diversify_candidates(self, candidates, n_recommendations=None, n_candidate_clusters=None):
        """Run the improved pipeline on the candidate pool and select the final rows."""
        if candidates.empty:
            return candidates

        n_recommendations, n_clusters = self._resolve_diversity_settings(
            candidates,
            n_recommendations,
            n_candidate_clusters,
        )
        if n_clusters is None:
            return candidates.head(n_recommendations).copy()

        clustered_candidates = self._cluster_candidates(candidates, n_clusters)
        grouped_candidates, cluster_priority = self._rank_clusters(clustered_candidates)

        return self._select_diverse_rows(
            grouped_candidates,
            cluster_priority,
            n_recommendations,
        )

    def recommend(
        self,
        movie_title,
        return_baseline=False,
        n_recommendations=None,
        candidate_pool_size=None,
        n_candidate_clusters=None,
    ):
    
        config = self.config.diversity
        n_recommendations = n_recommendations or config.n_recommendations
        candidate_pool_size = candidate_pool_size or config.candidate_pool_size

        candidates = self.get_candidate_movies(
            movie_title,
            candidate_pool_size=candidate_pool_size,
        )
        if candidates.empty:
            return candidates

        if return_baseline:
            return candidates.head(n_recommendations).copy()

        return self._diversify_candidates(
            candidates,
            n_recommendations=n_recommendations,
            n_candidate_clusters=n_candidate_clusters,
        )


# Mainly for development stage (reduce run time)
def build_recommender(
    data_dir=DATA_DIR,
    config=RecommendationConfig(),
    cache_path=DEFAULT_MOVIE_CACHE_PATH,
    use_cache=True,
    refresh_cache=False,
):
    """Run the preprocessing functions first, then fit the recommender estimator."""
    cache_file = Path(cache_path)

    if use_cache and cache_file.exists() and not refresh_cache:
        movie_table = pd.read_pickle(cache_file)
    else:
        movies, ratings, tags = load_data(data_dir)
        movies, ratings, tags = clean_data(movies, ratings, tags)
        movie_table = build_movie_table_from_tables(ratings, movies, tags)
        movie_table = apply_movie_filters(
            movie_table,
            require_rating_and_tag=config.filters.require_rating_and_tag,
        )

        if use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            movie_table.to_pickle(cache_file)

    recommender = MovieRecommender(config=config)
    recommender.fit(movie_table)
    return recommender
