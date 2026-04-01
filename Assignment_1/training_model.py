from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from config import RecommendationConfig

# Shared vectorizer for genres so the same
# definition is reused in both pipelines.
def build_genres_vectorizer(vectorizer_config):
    return TfidfVectorizer(
        min_df=vectorizer_config.min_df,
        max_df=vectorizer_config.max_df,
        max_features=vectorizer_config.max_features,
    )


def build_tag_vectorizer(vectorizer_config, n_documents=None):
    min_df = vectorizer_config.min_df
    max_df = vectorizer_config.max_df

    if n_documents is not None and n_documents > 0 and n_documents * max_df < min_df:
        min_df = 1
        max_df = 1.0

    return TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=vectorizer_config.max_features,
    )


# Shared feature blocks used by
# both the retrieval and clustering pipelines.
def build_base_feature_blocks(config=RecommendationConfig(), n_documents=None):
    return [
        (
            "genres",
            build_genres_vectorizer(config.genre_vectorizer),
            "genres_text",
        ),
        (
            "tags",
            build_tag_vectorizer(config.tag_vectorizer, n_documents=n_documents),
            "tag_text",
        ),
    ]


# Builds the shared feature matrix used by both retrieval and clustering.
def build_feature_transformer(config=RecommendationConfig(), n_documents=None):
    return ColumnTransformer(build_base_feature_blocks(config, n_documents=n_documents))


def build_baseline_pipeline(config=RecommendationConfig()):
    return Pipeline(
        [
            ("features", build_feature_transformer(config)),
            (
                "retriever",
                NearestNeighbors(
                    n_neighbors=config.candidate_pool.retrieval_pool_size + 1,
                    metric=config.neighbors.metric,
                    algorithm=config.neighbors.algorithm,
                ),
            ),
        ]
    )


def build_clustering_pipeline(config=RecommendationConfig(), n_clusters=5, n_documents=None):
    """Cluster the candidate pool in the shared feature space."""
    return Pipeline(
        [
            ("features", build_feature_transformer(config, n_documents=n_documents)),
            ("clusterer", KMeans(n_clusters=n_clusters, n_init=config.clustering.n_init)),
        ]
    )
