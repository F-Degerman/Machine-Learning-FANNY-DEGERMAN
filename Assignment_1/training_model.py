from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURE_COLUMNS = ["year_numeric", "mean_rating", "rating_count"]


# These dataclasses collect model settings in one place and
# make the pipeline configuration much easier to read and reuse.
@dataclass(frozen=True)
class MovieFilterConfig:
    require_rating_and_tag: bool = True


@dataclass(frozen=True)
class VectorizerConfig:
    min_df: int = 5
    max_df: float = 0.8
    max_features: int = 5000


@dataclass(frozen=True)
class DiversityConfig:
    candidate_pool_size: int = 20
    n_candidate_clusters: int = 5
    n_recommendations: int = 5


@dataclass(frozen=True)
class RecommendationConfig:
    filters: MovieFilterConfig = MovieFilterConfig()
    vectorizer: VectorizerConfig = VectorizerConfig()
    diversity: DiversityConfig = DiversityConfig()


# custom estimator for nearest-neighbor retrieval 
class NearestNeighborsRetriever(BaseEstimator):

    def __init__(self, n_neighbors=20, metric="cosine", algorithm="brute"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm

    def fit(self, X, y=None):
        self.model_ = NearestNeighbors(
            metric=self.metric,
            algorithm=self.algorithm,
            n_neighbors=self.n_neighbors,
        )
        self.model_.fit(X)
        return self

    def kneighbors(self, X, n_neighbors=None):
        return self.model_.kneighbors(X, n_neighbors=n_neighbors)


# shared vectorizer to keep the genre-processing logic in one place and
# to avoid repeating the same vectorizer definition twice.
def build_genres_vectorizer(vectorizer_config):
    return TfidfVectorizer(
        min_df=1,
        max_df=1.0,
        max_features=max(50, vectorizer_config.max_features // 10),
    )


# shared blocks to represent the common baseline feature space.
# Removes duplicated feature definitions.
def build_base_feature_blocks(config=RecommendationConfig()):
    vectorizer = config.vectorizer
    return [
        (
            "genres",
            build_genres_vectorizer(vectorizer),
            "genres_text",
        ),
        (
            "numeric",
            StandardScaler(),
            NUMERIC_FEATURE_COLUMNS,
        ),
    ]


# transformer for numeric/text feature matrix before fit of nearest-neighbor search
def build_similarity_transformer(config=RecommendationConfig()):
    return ColumnTransformer(build_base_feature_blocks(config))


# transformer so KMeans get same baseline features plus
#  tag text when clustering the candidate pool.
def build_candidate_feature_transformer(config=RecommendationConfig()):
    vectorizer = config.vectorizer

    return ColumnTransformer(
        [
            *build_base_feature_blocks(config),
            (
                "tags",
                TfidfVectorizer(
                    min_df=vectorizer.min_df,
                    max_df=vectorizer.max_df,
                    max_features=vectorizer.max_features,
                ),
                "tag_text",
            ),
        ]
    )


def build_baseline_pipeline(config=RecommendationConfig()):
    return Pipeline(
        [
            ("features", build_similarity_transformer(config)),
            (
                "retriever",
                NearestNeighborsRetriever(
                    n_neighbors=config.diversity.candidate_pool_size + 1,
                ),
            ),
        ]
    )


def build_improved_pipeline(config=RecommendationConfig(), n_clusters=5):
    """Improved pipeline: richer candidate features followed by KMeans clustering."""
    return Pipeline(
        [
            ("features", build_candidate_feature_transformer(config)),
            ("clusterer", KMeans(n_clusters=n_clusters, n_init=10)),
        ]
    )
