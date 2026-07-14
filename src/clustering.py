"""Segment products based on sales behaviour and promo response.

The feature set was built by starting with all behavioural variables, then removing highly correlated ones to avoid duplicate information

Several clustering methods were compared on the same preprocessed data. KMeans was chosen because it gave the most consistent and interpretable results. 
HDBSCAN was not stable on this dataset, and GMM did not provide any clear advantage.

UMAP followed by KMeans was also tested. It produced str
onger cluster separation, but the resulting clusters are much harder to explain from a business perspective. For that reason, the default approach remains 
KMeans on the original feature space, with the UMAP version kept as an alternative for comparison.
"""
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

UMAP_N_COMPONENTS = 5
UMAP_N_NEIGHBORS = 15

# non-negative, right-skewed (|skew| > 1.5) - plain log1p
SKEWED_NONNEG = ["avg_weekly_sales", "momentum", "mat_growth", "promo_lift", "promo_intensity_effect"]
# can go negative - signed log (sign(x) * log1p(abs(x))), plain log1p breaks on negative input
SKEWED_SIGNED = ["growth_rate", "acv_trend", "price_elasticity", "promo_effect_ctrl"]
# |skew| < 1.5, or binary (is_new/is_delisted - binary variables are never
# log-transformed regardless of their skew statistic, which just reflects
# class imbalance, not a distribution shape a log fixes)
UNSKEWED = ["avg_acv", "sales_cv", "promo_week_share", "promo_sales_share", "n_retailers", "avg_price", "is_new", "is_delisted", "peer_correlation", 
            "promo_visibility_share", "promo_lift_retailer_cv", "acv_retailer_cv", "substitute_correlation",]

FEATURE_COLS = SKEWED_NONNEG + SKEWED_SIGNED + UNSKEWED


def _signed_log1p(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def prep_matrix(feats: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    X = feats[FEATURE_COLS].copy()
    for col in SKEWED_NONNEG:
        X[col] = np.log1p(X[col].clip(lower=0))
    for col in SKEWED_SIGNED:
        X[col] = _signed_log1p(X[col])
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def choose_k(X: np.ndarray, k_range=range(2, 9), seed: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def fit(feats: pd.DataFrame, k: int, seed: int = 42):
    X, scaler = prep_matrix(feats)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    labels = pd.Series(km.labels_, index=feats.index, name="cluster")
    return labels, km, scaler


def reduce_umap(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """UMAP was tested because it can capture non-linear patterns that KMeans on the original feature space may miss. It produced more stable and better-separated 
    clusters, but at the cost of interpretability
    The main drawback is that clustering happens in an abstract embedding rather than on the original business features. Clusters can still be profiled using the 
    original variables, but the boundaries themselves are harder to explain
    """
    n_neighbors = min(UMAP_N_NEIGHBORS, X.shape[0] - 1)
    reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, n_neighbors=n_neighbors, random_state=seed)
    return reducer.fit_transform(X)


def fit_umap(feats: pd.DataFrame, k: int, seed: int = 42):
    X, scaler = prep_matrix(feats)
    X_umap = reduce_umap(X, seed=seed)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X_umap)
    labels = pd.Series(km.labels_, index=feats.index, name="cluster")
    return labels, km, scaler


def fit_umap_hdbscan(feats: pd.DataFrame, min_cluster_size: int, seed: int = 42):
    """Apply HDBSCAN on the UMAP embedding.

    Unlike HDBSCAN on the original feature space, the UMAP embedding produces stable clusters. However, the useful settings either collapse into very broad 
    groups or become unstable when more detailed clusters are requested.

    Overall, HDBSCAN did not provide a better balance of stability, granularity, and interpretability than KMeans, so it is included as a tested alternative 
    rather than the recommended approach.
    """
    X, scaler = prep_matrix(feats)
    X_umap = reduce_umap(X, seed=seed)
    model = HDBSCAN(min_cluster_size=min_cluster_size).fit(X_umap)
    labels = pd.Series(model.labels_, index=feats.index, name="cluster")
    return labels, model, scaler


def profile(feats: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Median of each raw (unscaled) feature per cluster + size of each cluster. Useful for profiling and naming clusters"""
    out = feats[FEATURE_COLS].join(labels).groupby("cluster").median()
    out["n_ppgs"] = labels.value_counts()
    return out.round(3)


def _bootstrap_ari(feats: pd.DataFrame, fit_fn, param, n_boot: int, frac: float, seed: int) -> float:
    """`param` is the second argument passed to `fit_fn` (for example, `k` or `min_cluster_size`). Since ARI is permutation-invariant, 
    cluster labels do not need to be aligned before comparison"""
    rng = np.random.RandomState(seed)
    full_labels, _, _ = fit_fn(feats, param, seed=seed)
    scores = []
    for i in range(n_boot):
        sample_idx = rng.choice(feats.index, size=int(len(feats) * frac), replace=False)
        boot_labels, _, _ = fit_fn(feats.loc[sample_idx], param, seed=seed + i + 1)
        scores.append(adjusted_rand_score(full_labels.loc[sample_idx], boot_labels))
    return float(np.mean(scores))


def stability_ari(feats: pd.DataFrame, k: int, n_boot: int = 20, frac: float = 0.85, seed: int = 42) -> float:
    """Estimate clustering stability by bootstrap resampling. Each resampled dataset is clustered again, and the result is compared with the original 
    clustering using Adjusted Rand Index (ARI). A low ARI suggests the clusters are not stable"""
    return _bootstrap_ari(feats, fit, k, n_boot, frac, seed)


def stability_ari_umap(feats: pd.DataFrame, k: int, n_boot: int = 20, frac: float = 0.85, seed: int = 42) -> float:
    """Same bootstrap stability check as `stability_ari`, but using UMAP + KMeans."""
    return _bootstrap_ari(feats, fit_umap, k, n_boot, frac, seed)


def stability_ari_umap_hdbscan(feats: pd.DataFrame, min_cluster_size: int, n_boot: int = 20, frac: float = 0.85, seed: int = 42) -> float:
    """Same bootstrap stability check, but using UMAP + HDBSCAN. Included for comparison, although it is not the recommended clustering method"""
    return _bootstrap_ari(feats, fit_umap_hdbscan, min_cluster_size, n_boot, frac, seed)


def flag_weak_fits(feats: pd.DataFrame, labels: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Calculate the silhouette score for each product. Negative values indicate products that fit better in another cluster and may be weak or ambiguous assignments"""
    X, _ = prep_matrix(feats)
    sil = silhouette_samples(X, labels.values)
    return pd.Series(sil, index=feats.index, name="silhouette").lt(threshold)


def flag_weak_fits_umap(feats: pd.DataFrame, labels: pd.Series, seed: int = 42, threshold: float = 0.0) -> pd.Series:
    """Calculate per-point silhouette scores in the UMAP embedding, using the same feature space where clustering was performed"""
    X, _ = prep_matrix(feats)
    X_umap = reduce_umap(X, seed=seed)
    sil = silhouette_samples(X_umap, labels.values)
    return pd.Series(sil, index=feats.index, name="silhouette").lt(threshold)


def representative_ppgs(feats: pd.DataFrame, labels: pd.Series, km, X: np.ndarray) -> dict:
    """Return the PPG closest to each cluster centroid. The input feature matrix must match the one used to fit the model"""
    reps = {}
    for cid in sorted(labels.unique()):
        idx = np.where(labels.values == cid)[0]
        centroid = km.cluster_centers_[cid]
        dists = np.linalg.norm(X[idx] - centroid, axis=1)
        reps[cid] = feats.index[idx[dists.argmin()]]
    return reps

def _map_cluster(pr_weekly: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Attach each row's cluster; drop rows for PPGs with no label (e.g.
    anomalies excluded before clustering, or a different label set than
    the one passed in)."""
    out = pr_weekly.assign(cluster=pr_weekly["rgm_ppg"].map(labels))
    return out.dropna(subset=["cluster"])

def segment_weekly_sales(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Cluster x week total sales, in currency"""
    df = _clean(df)
    pr_weekly = build_retailer_weekly_panel(df)
    mapped = _map_cluster(pr_weekly, labels)
    return (
        mapped.groupby(["period_id", "cluster"])["sales"].sum()
        .unstack("cluster").sort_index().fillna(0)
    )