"""PPG-level feature engineering. Raw weekly panel -> one row per rgm_ppg."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sstats

LIFECYCLE_GAP_WEEKS = 26
MIN_OBS_FOR_REGRESSION = 8
MOMENTUM_WINDOW_WEEKS = 12
MAT_WINDOW_WEEKS = 52
MIN_MONTHS_FOR_SEASONALITY = 6
WEEK_ANCHOR = "W-SAT"  # data is Saturday-anchored, not pandas' default Sunday
PEER_MIN_OVERLAP_WEEKS = 30  # min shared weeks before trusting a pair's correlation
PEER_ALPHA = 0.05  # significance level, Bonferroni-corrected for all pairs tested at once


def _slope(y: np.ndarray) -> float:
    if len(y) < 3 or np.all(y == y[0]):
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def _fill_gaps(g: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto a continuous weekly calendar; missing weeks = 0 activity."""
    full_range = pd.date_range(g["period_id"].min(), g["period_id"].max(), freq=WEEK_ANCHOR)
    out = g.set_index("period_id").reindex(full_range)
    zero_cols = ["units", "sales", "promo_units", "promo_amt", "acv", "promo_acv"]
    out[zero_cols] = out[zero_cols].fillna(0)
    return out.rename_axis("period_id")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_id"] = pd.to_datetime(df["period_id"])
    df["any_promo_units"] = df["any_promo_units"].clip(lower=0)
    df["any_promo_amt"] = df["any_promo_amt"].clip(lower=0, upper=df["total_sales"])
    df["price"] = df["total_sales"] / df["total_units"].replace(0, np.nan)
    df["on_promo"] = (df["any_promo_units"] > 0).astype(int)
    return df


def build_retailer_weekly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """ppg x retailer x week, gap-filled per ppg-retailer pair. Exposed
    (not underscore-prefixed) because interactions.py needs the
    retailer-level detail -- ppg-level totals would hide which retailer
    a co-movement pattern is actually happening in."""
    pr_weekly = (
        df.groupby(["rgm_ppg", "retailer_nm", "period_id"])
        .agg(
            units=("total_units", "sum"),
            sales=("total_sales", "sum"),
            promo_units=("any_promo_units", "sum"),
            promo_amt=("any_promo_amt", "sum"),
            acv=("acv_pct", "mean"),
            promo_acv=("any_promo_acv_pct", "mean"),
        )
        .reset_index()
        .sort_values(["rgm_ppg", "retailer_nm", "period_id"])
    )
    pr_weekly = (
        pr_weekly.groupby(["rgm_ppg", "retailer_nm"], group_keys=True)
        .apply(_fill_gaps, include_groups=False)
        .reset_index()
    )
    pr_weekly["on_promo"] = (pr_weekly["promo_units"] > 0).astype(int)
    return pr_weekly


def _build_weekly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Gap-fill per ppg-retailer, then sum retailers into one series per ppg."""
    pr_weekly = build_retailer_weekly_panel(df)
    weekly = (
        pr_weekly.groupby(["rgm_ppg", "period_id"])
        .agg(
            units=("units", "sum"),
            sales=("sales", "sum"),
            promo_units=("promo_units", "sum"),
            promo_amt=("promo_amt", "sum"),
            acv=("acv", "mean"),
            promo_acv=("promo_acv", "mean"),
        )
        .reset_index()
        .sort_values(["rgm_ppg", "period_id"])
    )
    weekly["on_promo"] = (weekly["promo_units"] > 0).astype(int)
    weekly["price"] = weekly["sales"] / weekly["units"].replace(0, np.nan)
    weekly["log_sales"] = np.log1p(weekly["sales"])
    weekly["log_price"] = np.log(weekly["price"].replace(0, np.nan))
    weekly["log_units"] = np.log1p(weekly["units"])
    return weekly


def _price_regression(g: pd.DataFrame) -> pd.Series:
    """Elasticity, controlling for promo weeks (price and promo both move together)."""
    valid = g[["log_units", "log_price", "on_promo"]].dropna()
    if len(valid) < MIN_OBS_FOR_REGRESSION or valid["log_price"].std() == 0:
        return pd.Series({"price_elasticity": np.nan, "promo_effect_ctrl": np.nan})
    X = sm.add_constant(valid[["log_price", "on_promo"]])
    model = sm.OLS(valid["log_units"], X).fit()
    return pd.Series({
        "price_elasticity": model.params["log_price"],
        "promo_effect_ctrl": model.params["on_promo"],
    })


def _promo_features(g: pd.DataFrame) -> pd.Series:
    promo_mask = g["on_promo"] == 1
    base_sales = g.loc[~promo_mask, "sales"]
    promo_sales = g.loc[promo_mask, "sales"]
    promo_lift = (
        promo_sales.mean() / base_sales.mean()
        if len(base_sales) and base_sales.mean()
        else np.nan
    )
    promo_sales_share = (
        (g["promo_amt"] / g["sales"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .mean()
    )
    promo_acv_share = (
        (g["promo_acv"] / g["acv"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .mean()
    )
    return pd.Series({
        "promo_week_share": promo_mask.mean(),
        "promo_sales_share": promo_sales_share,
        "promo_acv_share": promo_acv_share,
        "promo_lift": promo_lift,
    })


def _windowed_growth(g: pd.DataFrame, window: int) -> float:
    """Recent `window` weeks vs. the prior `window` weeks. Used for both
    momentum (12) and MAT growth (52, season-controlled)."""
    if len(g) < 2 * window:
        return np.nan
    recent = g["sales"].tail(window).mean()
    prior = g["sales"].iloc[-2 * window : -window].mean()
    return recent / prior if prior else np.nan


def _seasonality_index(g: pd.DataFrame) -> float:
    """Spread across calendar month vs. mean. Groups on dt.month (1-12),
    not month_nm, which is a year-month string."""
    month_avg = g.groupby(g["period_id"].dt.month)["sales"].mean()
    if len(month_avg) < MIN_MONTHS_FOR_SEASONALITY or month_avg.mean() == 0:
        return np.nan
    return (month_avg.max() - month_avg.min()) / month_avg.mean()


def _residuals_by_retailer(pr_weekly: pd.DataFrame) -> pd.DataFrame:
    """log-sales residual vs. each ppg-retailer's own non-promo baseline,
    within each retailer -- a PPG's pattern at one retailer shouldn't be
    contaminated by a different retailer's promo calendar."""
    out = pr_weekly.copy()
    out["log_sales"] = np.log1p(out["sales"])
    baseline = out.loc[out.on_promo == 0].groupby(["rgm_ppg", "retailer_nm"])["log_sales"].mean()
    out = out.set_index(["rgm_ppg", "retailer_nm"])
    out["baseline"] = baseline
    out = out.reset_index()
    out["residual"] = out["log_sales"] - out["baseline"]
    market = out.groupby(["retailer_nm", "period_id"])["residual"].transform("median")
    out["residual"] = out["residual"] - market  # remove shared retailer-week calendar effect
    return out


def _peer_correlation_matrix(pr_weekly_resid: pd.DataFrame, min_overlap: int = PEER_MIN_OVERLAP_WEEKS):
    """PPG x PPG correlation of residuals, combined across retailers via
    Fisher's z (correct way to average correlations -- naive averaging
    understates estimates built on more observations). Returns (corr,
    effective_n); effective_n feeds the significance test below."""
    all_ppgs = sorted(pr_weekly_resid["rgm_ppg"].unique())
    z_mats, n_mats = [], []
    for _, g in pr_weekly_resid.groupby("retailer_nm"):
        wide = g.pivot(index="period_id", columns="rgm_ppg", values="residual")
        corr = wide.corr(min_periods=min_overlap).reindex(index=all_ppgs, columns=all_ppgs)
        overlap = wide.notna().astype(int)
        n_obs = overlap.T.dot(overlap).reindex(index=all_ppgs, columns=all_ppgs)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.arctanh(corr.values.clip(-0.999, 0.999))
        z[n_obs.values < min_overlap] = np.nan
        z_mats.append(z)
        n_mats.append(n_obs.values.astype(float))
    weights = np.where(np.isnan(np.stack(z_mats)), 0, np.stack(n_mats) - 3)
    weights = np.clip(weights, 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_weighted = np.nansum(np.stack(z_mats) * weights, axis=0) / np.where(weights.sum(axis=0) == 0, np.nan, weights.sum(axis=0))
    combined = np.tanh(z_weighted)
    return (
        pd.DataFrame(combined, index=all_ppgs, columns=all_ppgs),
        pd.DataFrame(weights.sum(axis=0), index=all_ppgs, columns=all_ppgs),
    )


def _significant_pairs_mask(corr: pd.DataFrame, effective_n: pd.DataFrame, alpha: float = PEER_ALPHA) -> pd.DataFrame:
    """Bonferroni-corrected: with n*(n-1)/2 pairs tested at once, alpha
    gets divided by that count first -- a raw correlation threshold alone
    would flag many spurious pairs by chance at this scale."""
    n_pairs = (corr.shape[0] * (corr.shape[0] - 1)) // 2
    if n_pairs == 0:
        return pd.DataFrame(False, index=corr.index, columns=corr.columns)
    alpha_corrected = alpha / n_pairs
    df = np.clip(effective_n.values - 2, 1, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = corr.values * np.sqrt(df / (1 - corr.values ** 2))
    p = 2 * (1 - sstats.t.cdf(np.abs(t_stat), df=df))
    return pd.DataFrame(p < alpha_corrected, index=corr.index, columns=corr.columns)


def _peer_correlation(df: pd.DataFrame, brand: pd.Series) -> pd.Series:
    """Each PPG's average significant correlation with same-BRAND peers.
    Brand, not subcategory -- checked empirically, brand gives more spread
    (std 0.37 vs 0.33) and is the conceptually tighter peer group (a
    manufacturer's own line, not a whole subcategory full of competitors)."""
    pr_weekly = build_retailer_weekly_panel(df)
    resid = _residuals_by_retailer(pr_weekly)
    corr, n_eff = _peer_correlation_matrix(resid)
    mask = _significant_pairs_mask(corr, n_eff)
    masked = corr.where(mask)
    groups = brand.reindex(corr.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.arctanh(masked.values.clip(-0.999, 0.999))
    out = pd.Series(index=corr.index, dtype=float, name="peer_correlation")
    for i, ppg in enumerate(corr.index):
        same_group = np.where((groups.values == groups.loc[ppg]) & (corr.index != ppg))[0]
        vals = z[i, same_group]
        valid = vals[~np.isnan(vals)]
        out.loc[ppg] = np.tanh(valid.mean()) if len(valid) else np.nan
    return out


def _compute_features(weekly: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    global_min, global_max = df["period_id"].min(), df["period_id"].max()
    grouped = weekly.groupby("rgm_ppg")

    level = grouped.agg(
        n_weeks=("sales", "size"),
        first_seen=("period_id", "min"),
        last_seen=("period_id", "max"),
        avg_weekly_sales=("sales", "mean"),
        avg_weekly_units=("units", "mean"),
        sales_std=("sales", "std"),
        avg_acv=("acv", "mean"),
    )
    level["sales_cv"] = level["sales_std"] / level["avg_weekly_sales"]
    level["is_new"] = ((level["first_seen"] - global_min).dt.days > LIFECYCLE_GAP_WEEKS * 7).astype(int)
    level["is_delisted"] = ((global_max - level["last_seen"]).dt.days > LIFECYCLE_GAP_WEEKS * 7).astype(int)

    growth = grouped["log_sales"].apply(lambda s: _slope(s.to_numpy())).rename("growth_rate")
    acv_trend = grouped["acv"].apply(lambda s: _slope(s.to_numpy())).rename("acv_trend")
    momentum = grouped.apply(_windowed_growth, window=MOMENTUM_WINDOW_WEEKS, include_groups=False).rename("momentum")
    mat_growth = grouped.apply(_windowed_growth, window=MAT_WINDOW_WEEKS, include_groups=False).rename("mat_growth")
    seasonality = grouped.apply(_seasonality_index, include_groups=False).rename("seasonality_index")
    promo = grouped.apply(_promo_features, include_groups=False)
    price_reg = grouped.apply(_price_regression, include_groups=False)
    avg_price = grouped["price"].mean().rename("avg_price")
    n_retailers = df.groupby("rgm_ppg")["retailer_nm"].nunique().rename("n_retailers")

    feats = (
        level.drop(columns=["first_seen", "last_seen", "sales_std"])
        .join(growth)
        .join(acv_trend)
        .join(momentum)
        .join(mat_growth)
        .join(seasonality)
        .join(promo)
        .join(price_reg)
        .join(avg_price)
        .join(n_retailers)
    )

    # median-fill + flag rather than drop rows; fall back to 0 if a whole column is NaN
    fillable = ["price_elasticity", "promo_effect_ctrl", "promo_lift", "momentum", "mat_growth", "seasonality_index"]
    for col in fillable:
        #feats[col + "_missing"] = feats[col].isna().astype(int)
        feats[col] = feats[col].fillna(feats[col].median()).fillna(0)

    meta = (
        df.drop_duplicates("rgm_ppg")
        .set_index("rgm_ppg")[[
            "is_own_manufacturer", "subcategory_nm", "brand_nm",
            "attribute_9", "attribute_1", "attribute_7",
        ]]
        .rename(columns={"attribute_9": "price_tier", "attribute_1": "form", "attribute_7": "benefit_type"})
    )
    feats = feats.join(meta)

    # within-PPG interaction terms -- built on the same log/signed-log scale
    # each parent gets before clustering, not raw values: raw promo_week_share
    # * promo_lift came out 0.95 correlated with promo_lift alone (its huge
    # outlier-driven variance swamps the bounded 0-1 factor)
    feats["promo_intensity_effect"] = feats["promo_week_share"] * np.log1p(feats["promo_lift"])
    feats["dist_trend"] = feats["avg_acv"] * (np.sign(feats["growth_rate"]) * np.log1p(np.abs(feats["growth_rate"])))

    # peer_correlation: average residual-sales correlation with same-BRAND
    # peers, using the same significance-tested correlation logic above --
    # not a separate module, computed right here.
    peer_corr = _peer_correlation(df, feats["brand_nm"])
    feats["peer_correlation_missing"] = peer_corr.reindex(feats.index).isna().astype(int)
    feats["peer_correlation"] = peer_corr.reindex(feats.index).fillna(peer_corr.median()).fillna(0)

    return feats


def build_ppg_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean(df)
    weekly = _build_weekly_panel(df)
    return _compute_features(weekly, df)


def assign_ppg_id(feats: pd.DataFrame) -> pd.DataFrame:
    """Short stable id for readability (charts, profile tables); keeps
    rgm_ppg as a column so it's still traceable."""
    out = feats.sort_index().copy()
    out.insert(0, "rgm_ppg", out.index)
    out.index = [f"PPG_{i+1:03d}" for i in range(len(out))]
    out.index.name = "ppg_id"
    return out


# Scaling: _clean and the groupBy/agg steps translate directly to PySpark.
# _fill_gaps and _price_regression run per-PPG-group and would need
# groupBy(...).applyInPandas(...) to distribute.
