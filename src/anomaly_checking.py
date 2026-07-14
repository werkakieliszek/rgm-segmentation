"""Find PPGs that look unusual using the same feature space as KMeans.

Isolation Forest is the only method used. LOF is also density-based, like HDBSCAN, which did not perform well on this dataset (18 features, 366 PPGs) 
because the data is too sparse for reliable density estimates. Elliptic Envelope assumes a multivariate Gaussian distribution, which is not suitable 
here due to the mix of binary features (`is_new`, `is_delisted`) and skewed continuous variables. Isolation Forest does not make these assumptions and 
is better suited for this data.

"""
import pandas as pd
from sklearn.ensemble import IsolationForest

from clustering import FEATURE_COLS, prep_matrix

DEFAULT_CONTAMINATION = 0.05


def flag_anomalies(feats: pd.DataFrame, contamination: float = DEFAULT_CONTAMINATION, seed: int = 42) -> pd.DataFrame:
    X, _ = prep_matrix(feats)
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=300)
    model.fit(X)
    out = pd.DataFrame(index=feats.index)
    out["anomaly_score"] = -model.score_samples(X)
    out["is_anomaly"] = model.predict(X) == -1
    return out.sort_values("anomaly_score", ascending=False)


def top_drivers(feats: pd.DataFrame, ppg_id, n: int = 3) -> pd.Series:
    """Which features are most extreme (highest |z-score|) for one PPG,
    explains why it was flagged, not just that it was."""
    z = (feats[FEATURE_COLS] - feats[FEATURE_COLS].mean()) / feats[FEATURE_COLS].std()
    return z.loc[ppg_id].abs().sort_values(ascending=False).head(n)
