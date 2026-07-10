"""PPG-level feature engineering. Raw weekly panel -> one row per rgm_ppg."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

LIFECYCLE_GAP_WEEKS = 26
MIN_OBS_FOR_REGRESSION = 8
MOMENTUM_WINDOW_WEEKS = 12
MAT_WINDOW_WEEKS = 52
MIN_MONTHS_FOR_SEASONALITY = 6
WEEK_ANCHOR = "W-SAT"  # data is Saturday-anchored, not pandas' default Sunday


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
    #df["any_promo_units"] = df["any_promo_units"].clip(lower=0)
    #df["any_promo_amt"] = df["any_promo_amt"].clip(lower=0, upper=df["total_sales"])
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
        feats[col + "_missing"] = feats[col].isna().astype(int)
        feats[col] = feats[col].fillna(feats[col].median()).fillna(0)

    meta = (
        df.drop_duplicates("rgm_ppg")
        .set_index("rgm_ppg")[[
            "is_own_manufacturer", "subcategory_nm", "brand_nm",
            "attribute_9", "attribute_1", "attribute_7", "product_unit_size",
        ]]
        .rename(columns={"attribute_9": "price_tier", "attribute_1": "form", "attribute_7": "benefit_type", "product_unit_size": "pack_size"})
    )
    return feats.join(meta)


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
