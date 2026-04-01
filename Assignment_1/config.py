from dataclasses import dataclass, field


@dataclass(frozen=True)
class TagVectorizerConfig:
    min_df: int = 5
    max_df: float = 0.8
    max_features: int = 5000


@dataclass(frozen=True)
class GenreVectorizerConfig:
    min_df: int = 1
    max_df: float = 1.0
    max_features: int = 500


@dataclass(frozen=True)
class DataConfig:
    require_rating_and_tag: bool = True


@dataclass(frozen=True)
class CandidatePoolConfig:
    retrieval_pool_size: int = 50
    candidate_pool_size: int = 20
    min_candidate_rating: float = 2.0
    min_candidate_rating_count: int = 20


@dataclass(frozen=True)
class DiversificationConfig:
    n_candidate_clusters: int = 5
    n_recommendations: int = 5


@dataclass(frozen=True)
class NeighborConfig:
    metric: str = "cosine"
    algorithm: str = "brute"


@dataclass(frozen=True)
class ClusteringConfig:
    n_init: int = 10


@dataclass(frozen=True)
class RankingConfig:
    similarity_weight: float = 0.5
    rating_weight: float = 0.5


@dataclass(frozen=True)
class MatchingConfig:
    fuzzy_match_cutoff: float = 0.75
    fuzzy_match_candidates: int = 5


@dataclass(frozen=True)
class RecommendationConfig:
    data: DataConfig = field(default_factory=DataConfig)
    genre_vectorizer: GenreVectorizerConfig = field(default_factory=GenreVectorizerConfig)
    tag_vectorizer: TagVectorizerConfig = field(default_factory=TagVectorizerConfig)
    candidate_pool: CandidatePoolConfig = field(default_factory=CandidatePoolConfig)
    diversification: DiversificationConfig = field(default_factory=DiversificationConfig)
    neighbors: NeighborConfig = field(default_factory=NeighborConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
