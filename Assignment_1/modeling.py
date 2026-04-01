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
    return TfidfVectorizer(
        min_df=vectorizer_config.min_df,
        max_df=vectorizer_config.max_df,
        max_features=vectorizer_config.max_features,
    )


# Candidate pools are small enough that strict tag pruning can remove all terms.
# Use a more permissive tag vectorizer for clustering so the feature space stays usable.
def build_candidate_tag_vectorizer(vectorizer_config):
    return TfidfVectorizer(
        min_df=1,
        max_df=1.0,
        max_features=vectorizer_config.max_features,
    )


# Shared feature blocks used by
# both the retrieval and clustering pipelines.
def build_base_feature_blocks(config=RecommendationConfig(), use_candidate_tag_vectorizer=False):
    tag_vectorizer = (
        build_candidate_tag_vectorizer(config.tag_vectorizer)
        if use_candidate_tag_vectorizer
        else build_tag_vectorizer(config.tag_vectorizer)
    )
    return [
        (
            "genres",
            build_genres_vectorizer(config.genre_vectorizer),
            "genres_text",
        ),
        (
            "tags",
            tag_vectorizer,
            "tag_text",
        ),
    ]


# Builds the shared feature matrix used by both retrieval and clustering.
def build_feature_transformer(config=RecommendationConfig(), use_candidate_tag_vectorizer=False):
    return ColumnTransformer(
        build_base_feature_blocks(
            config,
            use_candidate_tag_vectorizer=use_candidate_tag_vectorizer,
        )
    )


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

# Cluster the candidate pool in the shared feature space.
def build_clustering_pipeline(config=RecommendationConfig(), n_clusters=5, n_documents=None):
    return Pipeline(
        [
            ("features", build_feature_transformer(config, use_candidate_tag_vectorizer=True)),
            ("clusterer", KMeans(n_clusters=n_clusters, n_init=config.clustering.n_init)),
        ]
    )
