from difflib import get_close_matches

import pandas as pd
from sklearn.base import BaseEstimator

from config import RecommendationConfig
from movie_data import (
    DEFAULT_MOVIE_CACHE_PATH,
    DATA_DIR,
    build_movie_table,
)
from training_model import (
    build_baseline_pipeline,
    build_clustering_pipeline,
)

MATCH_RANK_COLUMNS = ["rating_count", "mean_rating"]
DEFAULT_RANK_COLUMNS = ["combined_score", "similarity_score", "rating_count"]
DEFAULT_RANK_ASCENDING = [False, False, False]

# Blend similarity with normalized rating so strong matches with better ratings rise slightly.
def add_combined_score(candidates, ranking_config):
    candidates = candidates.copy()
    rating_span = candidates["mean_rating"].max() - candidates["mean_rating"].min()
    # Normalize ratings within the current candidate pool.
    # If all candidates have the same mean_rating, rating cannot separate them,
    # so rating_score falls back to 0.0 for all rows.
    if rating_span > 0:
        candidates["rating_score"] = (
            (candidates["mean_rating"] - candidates["mean_rating"].min()) / rating_span
        )
    else:
        candidates["rating_score"] = 0.0

    candidates["combined_score"] = (
        ranking_config.similarity_weight * candidates["similarity_score"] +
        ranking_config.rating_weight * candidates["rating_score"]
    )
    return candidates


def rank_candidate_rows(candidates, ranking_config):
    candidates = add_combined_score(candidates, ranking_config)
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


# Keeps the full recommendation flow in one coherent model object
# instead of loose functions.
class MovieRecommender(BaseEstimator):

    def __init__(self, config=RecommendationConfig()):
        self.config = config

    def fit(self, movie_table, y=None):
        self.movie_table_ = movie_table.reset_index(drop=True).copy()
        self.movie_table_["normalized_title"] = self.movie_table_["clean_title"].map(normalize_title)

        self.baseline_pipeline_ = build_baseline_pipeline(self.config)
        self.baseline_pipeline_.fit(self.movie_table_)
        return self

    # Resolve the user's title string to the best-matching movie row in the fitted table.
    def resolve_movie_index(self, movie_title):
        normalized_query = normalize_title(movie_title)

        exact_matches = self.movie_table_[self.movie_table_["normalized_title"] == normalized_query]
        if not exact_matches.empty:
            return select_best_match(exact_matches)

        partial_matches = self.movie_table_[
            self.movie_table_["normalized_title"].str.contains(normalized_query, regex=False)
        ]
        if not partial_matches.empty:
            return select_best_match(partial_matches)

        fuzzy_titles = get_close_matches(
            normalized_query,
            self.movie_table_["normalized_title"].tolist(),
            n=self.config.matching.fuzzy_match_candidates,
            cutoff=self.config.matching.fuzzy_match_cutoff,
        )
        if fuzzy_titles:
            fuzzy_matches = self.movie_table_[self.movie_table_["normalized_title"].isin(fuzzy_titles)]
            return select_best_match(fuzzy_matches)

        raise ValueError(f"No movie title match found for '{movie_title}'.")

    # To keep the candidate-building loop shorter
    # and avoid repeating the same row-to-dict mapping logic inline.
    def build_candidate_row(self, movie_index, distance):
        movie_row = self.movie_table_.iloc[movie_index]
        return {
            "movieId": movie_row["movieId"],
            "title": movie_row["title"],
            "genres_text": movie_row["genres_text"],
            "tag_text": movie_row["tag_text"],
            "mean_rating": movie_row["mean_rating"],
            "rating_count": movie_row["rating_count"],
            "similarity_score": 1 - float(distance),
        }

    # Keeps broad title matching while filtering recommendation candidates for basic quality.
    def filter_candidate_pool_by_quality(self, candidates, n_recommendations):
        quality_filtered = candidates[
            (candidates["mean_rating"] >= self.config.candidate_pool.min_candidate_rating)
            & (candidates["rating_count"] >= self.config.candidate_pool.min_candidate_rating_count)
        ]
        if len(quality_filtered) < n_recommendations:
            return candidates

        return quality_filtered

    # Uses the retrieval model to fetch the larger pool before quality filtering and clustering.
    def retrieve_candidate_movies(self, movie_title, retrieval_pool_size=None):
        query_index = self.resolve_movie_index(movie_title)
        query_row = self.movie_table_.iloc[[query_index]]
        query_features = self.baseline_pipeline_.named_steps["features"].transform(query_row)

        requested_neighbors = retrieval_pool_size or self.config.candidate_pool.retrieval_pool_size
        n_neighbors = min(requested_neighbors + 1, len(self.movie_table_))
        distances, indices = self.baseline_pipeline_.named_steps["retriever"].kneighbors(
            query_features,
            n_neighbors=n_neighbors,
        )

        candidates = pd.DataFrame(
            [
                self.build_candidate_row(index, distance)
                for distance, index in zip(distances[0], indices[0])
                if index != query_index
            ]
        )
        if candidates.empty:
            return candidates

        return rank_candidate_rows(candidates, self.config.ranking)

    def prepare_candidate_pool(self, movie_title, n_recommendations, candidate_pool_size):
        candidates = self.retrieve_candidate_movies(
            movie_title,
            retrieval_pool_size=max(candidate_pool_size, self.config.candidate_pool.retrieval_pool_size),
        )
        if candidates.empty:
            return candidates

        candidates = self.filter_candidate_pool_by_quality(candidates, n_recommendations)
        candidates = rank_candidate_rows(candidates, self.config.ranking)
        return candidates.head(candidate_pool_size).reset_index(drop=True)

    # Selects the top-ranked movie from each cluster first,
    # if a cluster is empty; fills that slot from the globally best leftovers.
    def select_diverse_rows(self, grouped_candidates, cluster_priority, n_recommendations):
        selected_rows = []
        selected_movie_ids = set()

        for cluster_id in cluster_priority:
            cluster_rows = grouped_candidates[cluster_id]
            if cluster_rows.empty:
                continue

            top_row = cluster_rows.iloc[0].to_dict()
            selected_rows.append(top_row)
            selected_movie_ids.add(top_row["movieId"])

            if len(selected_rows) == n_recommendations:
                return pd.DataFrame(selected_rows)

        remaining_rows = [
            row.to_dict()
            for cluster_id in cluster_priority
            for _, row in grouped_candidates[cluster_id].iloc[1:].iterrows()
            if row["movieId"] not in selected_movie_ids
        ]

        if remaining_rows:
            remaining_candidates = rank_candidate_rows(pd.DataFrame(remaining_rows), self.config.ranking)
            needed = n_recommendations - len(selected_rows)
            selected_rows.extend(remaining_candidates.head(needed).to_dict("records"))

        return pd.DataFrame(selected_rows)

    # Resolves fallback values before clustering (if very few candidates) 
    def resolve_diversity_settings(self, candidates, n_recommendations, n_candidate_clusters):
        config = self.config.diversification
        n_recommendations = n_recommendations or config.n_recommendations
        n_candidate_clusters = n_candidate_clusters or config.n_candidate_clusters

        if len(candidates) <= n_recommendations:
            return n_recommendations, None

        n_clusters = min(n_candidate_clusters, n_recommendations, len(candidates))
        if n_clusters < 2:
            return n_recommendations, None

        return n_recommendations, n_clusters

    # applies KMeans and attaches cluster labels before
    # the final per-cluster selection step.
    def cluster_candidates(self, candidates, n_clusters):
        clustering_pipeline = build_clustering_pipeline(
            self.config,
            n_clusters=n_clusters,
            n_documents=len(candidates),
        )
        clustering_pipeline.fit(candidates)
        cluster_labels = clustering_pipeline.named_steps["clusterer"].labels_

        clustered_candidates = candidates.copy()
        clustered_candidates["cluster"] = cluster_labels
        return clustered_candidates.sort_values(
            by=["cluster", *DEFAULT_RANK_COLUMNS],
            ascending=[True, *DEFAULT_RANK_ASCENDING],
        )

    # Prioritizes clusters by the strongest top candidate
    # before the strict top-1-per-cluster selection.
    def rank_clusters(self, clustered_candidates):
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

    # Runs clustering on the prepared candidate pool
    # and returns the final rows.
    def diversify_candidates(self, candidates, n_recommendations=None, n_candidate_clusters=None):
        if candidates.empty:
            return candidates

        n_recommendations, n_clusters = self.resolve_diversity_settings(
            candidates,
            n_recommendations,
            n_candidate_clusters,
        )
        if n_clusters is None:
            return candidates.head(n_recommendations).copy()

        clustered_candidates = self.cluster_candidates(candidates, n_clusters)
        grouped_candidates, cluster_priority = self.rank_clusters(clustered_candidates)

        return self.select_diverse_rows(
            grouped_candidates,
            cluster_priority,
            n_recommendations,
        )

    def recommend(
        self,
        movie_title,
        return_reranked_candidates=False,
        n_recommendations=None,
        candidate_pool_size=None,
        n_candidate_clusters=None,
    ):
        pool_config = self.config.candidate_pool
        diversification_config = self.config.diversification
        n_recommendations = n_recommendations or diversification_config.n_recommendations
        candidate_pool_size = candidate_pool_size or pool_config.candidate_pool_size

        candidates = self.prepare_candidate_pool(
            movie_title,
            n_recommendations,
            candidate_pool_size,
        )
        if candidates.empty:
            return candidates

        if return_reranked_candidates:
            return candidates.head(n_recommendations).copy()

        return self.diversify_candidates(
            candidates,
            n_recommendations=n_recommendations,
            n_candidate_clusters=n_candidate_clusters,
        )


# Builds the fitted recommender and
# optionally reuses the cached movie table.
def build_recommender(
    data_dir=DATA_DIR,
    config=RecommendationConfig(),
    cache_path=DEFAULT_MOVIE_CACHE_PATH,
    use_cache=True,
    refresh_cache=False,
):
    # Reuse the shared movie-table builder so preprocessing and caching
    # stay defined in one place.
    movie_table = build_movie_table(
        data_dir=data_dir,
        require_rating_and_tag=config.data.require_rating_and_tag,
        cache_path=cache_path,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )

    recommender = MovieRecommender(config=config)
    recommender.fit(movie_table)
    return recommender
