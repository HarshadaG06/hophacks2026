"""Unsupervised exploration: lifespan view concentration + burstiness.

Optimized with Polars for high-performance data processing, feature scaling,
crosstab calculations, and fast Parquet IO.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed

from loader import PROCESSED, RAW, get_snapshot_date, load_videos, processed_path
from s5_burstiness import KLEINBERG_GAMMA, KLEINBERG_S, kleinberg_sensitivity_row

# Tunables
K_RANGE = range(2, 7)
RANDOM_STATE = 42
SENSITIVITY_N = 100
SENSITIVITY_GAMMAS = (0.5, 1.0, 2.0)
SENSITIVITY_S = (2, 3)
# Cap events for sensitivity reruns only — keeps Kleinberg calls tractable.
SENSITIVITY_MAX_EVENTS = 3000
SENSITIVITY_JOBS = -1

LIFESPAN_FEATURES = [
    "views_centroid_u",
    "peak_u_views",
    "share_views_first25",
    "top_video_centroid_u",
    "log_lifespan_days",
]
BURST_FEATURES = [
    "kleinberg_max_level",
    "top_burst_length_share",
    "share_of_views_in_bursts",
]
FEATURES = LIFESPAN_FEATURES + BURST_FEATURES

LIFESPAN_CORR_COLS = ["views_centroid_u", "top_video_centroid_u", "share_views_first25"]
BURST_CORR_COLS = [
    "kleinberg_max_level",
    "share_of_views_in_bursts",
    "burst_views_centroid_u",
    "views_burst_lift",
    "top_video_in_burst_share",
]


def _gmm_bic(name: str, values: np.ndarray, x_range: tuple[float, float] | None = None) -> str:
    x = values[np.isfinite(values)].reshape(-1, 1)
    if len(x) < 10:
        print(f"{name}: too few points for GMM")
        return "n/a"
    bic1 = GaussianMixture(n_components=1, random_state=RANDOM_STATE).fit(x).bic(x)
    bic2 = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(x).bic(x)
    winner = "2-component" if bic2 < bic1 else "1-component"
    print(f"{name}: BIC_1={bic1:.1f}  BIC_2={bic2:.1f}  -> {winner} wins")
    if x_range:
        counts, _ = np.histogram(x.ravel(), bins=20, range=x_range)
        print(f"  histogram: {counts.tolist()}")
    return winner


def _cluster(features_df: pl.DataFrame, cols: list[str], k: int) -> np.ndarray:
    """Standardizes features and fits KMeans clustering returning labels array."""
    # Filter valid rows in Polars first
    valid_df = features_df.select(cols)
    mask = valid_df.select(
        pl.all_horizontal(pl.col("*").is_not_null() & ~pl.col("*").is_nan())
    ).to_series().to_numpy()

    labels = np.full(len(features_df), np.nan, dtype=float)
    if mask.sum() < k:
        return labels

    X_mat = valid_df.filter(
        pl.all_horizontal(pl.col("*").is_not_null() & ~pl.col("*").is_nan())
    ).to_numpy()

    scaled = StandardScaler().fit_transform(X_mat)
    
    # Additional guard against zero-variance NaNs post-scaling
    finite_mask = np.isfinite(scaled).all(axis=1)
    if finite_mask.sum() < k:
        return labels

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    
    # Assign labels to valid finite rows
    valid_indices = np.where(mask)[0][finite_mask]
    labels[valid_indices] = km.fit_predict(scaled[finite_mask])
    
    return labels


def _print_histogram(name: str, series: pl.Series, bins: int = 20, x_range=None) -> None:
    v = series.drop_nulls().to_numpy()
    if len(v) == 0:
        print(f"{name}: no data")
        return
    counts, _ = np.histogram(v, bins=bins, range=x_range)
    print(f"{name}: n={len(v)}  hist={counts.tolist()}")


def _sensitivity_analysis(
    per_audio_pl: pl.DataFrame,
    videos_pd: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    music_ids = per_audio_pl["music_id"].to_list()
    sample_ids = rng.choice(
        music_ids,
        size=min(SENSITIVITY_N, len(music_ids)),
        replace=False,
    )
    
    # Group videos for quick lookup by sampled music_ids
    sample_set = set(sample_ids)
    groups = {mid: g for mid, g in videos_pd.groupby("music_id") if mid in sample_set}

    tasks = [
        (mid, s, gamma)
        for mid in sample_ids
        for s, gamma in itertools.product(SENSITIVITY_S, SENSITIVITY_GAMMAS)
        if mid in groups
    ]
    print(
        f"\n=== Sensitivity ({len(sample_ids)} audios, {len(tasks)} Kleinberg runs, "
        f"max_events={SENSITIVITY_MAX_EVENTS}) ==="
    )
    t_sens = time.perf_counter()
    setting_rows = Parallel(n_jobs=SENSITIVITY_JOBS)(
        delayed(kleinberg_sensitivity_row)(
            mid, groups[mid], snapshot, s, gamma, SENSITIVITY_MAX_EVENTS
        )
        for mid, s, gamma in tasks
    )
    print(f"  sensitivity wall time: {time.perf_counter() - t_sens:.1f}s")

    if not setting_rows:
        return {}

    sens_pl = pl.DataFrame(setting_rows)
    if sens_pl.is_empty():
        return {}

    # Spearman correlation of each metric across parameter settings
    corrs = {}
    for col in ("kleinberg_max_level", "share_of_views_in_bursts"):
        # Pivot wide via Polars
        wide = sens_pl.pivot(
            on=["s", "gamma"],
            index="music_id",
            values=col
        )
        
        # Convert to numpy matrix to perform pairwise Spearman rank stability
        feature_cols = [c for c in wide.columns if c != "music_id"]
        matrix = wide.select(feature_cols).to_numpy()
        
        rhos = []
        n_cols = matrix.shape[1]
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                a = matrix[:, i]
                b = matrix[:, j]
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() >= 5:
                    rho, _ = spearmanr(a[mask], b[mask])
                    rhos.append(float(rho))
                    
        med = float(np.median(rhos)) if rhos else float("nan")
        corrs[col] = med
        print(f"  median Spearman {col} across settings: {med:.3f} (n_pairs={len(rhos)})")

    return corrs


def main(k: int | None = None) -> None:
    t0 = time.perf_counter()
    
    # Load features using Polars
    features_pl = pl.read_parquet(processed_path("lifespan_per_audio.parquet"))
    features_pl = features_pl.with_columns(
        pl.col("music_id").cast(pl.Utf8),
        (pl.col("lifespan_days").cast(pl.Float64) + 1.0).log().alias("log_lifespan_days")
    )

    # Topic/Sentiment Integration
    topics_path = RAW / "topics.parquet"
    if not topics_path.exists():
        topics_path = PROCESSED / "topics.parquet"
        
    if topics_path.exists():
        topics_pl = pl.read_parquet(topics_path).with_columns(pl.col("music_id").cast(pl.Utf8))
        features_pl = features_pl.join(topics_pl, on="music_id", how="left")
        print(f"Merged topics from {topics_path}")
    else:
        print("No topics.parquet found — skipping topic/sentiment merge")

    # 1. Histograms + GMM
    print("\n=== Kleinberg histogram ===")
    _print_histogram("kleinberg_max_level", features_pl["kleinberg_max_level"])

    print("\n=== Bimodality (GMM BIC) ===")
    views_centroid_arr = features_pl["views_centroid_u"].drop_nulls().to_numpy()
    bimodal_views = _gmm_bic("views_centroid_u", views_centroid_arr, (0, 1))

    # 2. Relationship checks
    print("\n=== views_centroid_u vs burst_views_centroid_u ===")
    sub_pl = features_pl.select(["views_centroid_u", "burst_views_centroid_u"]).drop_nulls()
    if len(sub_pl) >= 5:
        v_cent = sub_pl["views_centroid_u"].to_numpy()
        b_cent = sub_pl["burst_views_centroid_u"].to_numpy()
        rho, p = spearmanr(v_cent, b_cent)
        print(f"  Spearman rho={rho:.3f}  p={p:.2e}  n={len(sub_pl)}")
        
        # Display summary statistics via pandas conversion for readable printout
        print(sub_pl.to_pandas().describe().to_string())

    print("\n=== Spearman: burst features vs lifespan features ===")
    corr_cols = [c for c in BURST_CORR_COLS + LIFESPAN_CORR_COLS if c in features_pl.columns]
    
    # Calculate Spearman correlation matrix using pandas on extracted matrix
    sub_corr_pd = features_pl.select(corr_cols).to_pandas()
    corr_df = sub_corr_pd.corr(method="spearman")
    
    burst_part = corr_df.loc[
        [c for c in BURST_CORR_COLS if c in corr_df.index],
        [c for c in LIFESPAN_CORR_COLS if c in corr_df.columns],
    ]
    print(burst_part.round(3).to_string())

    # 3. Burst membership vs concentration
    print("\n=== Does burst membership explain concentration? ===")
    lift_series = features_pl["views_burst_lift"].drop_nulls()
    if len(lift_series) > 0:
        med_lift = lift_series.median()
        share_above_1 = (lift_series > 1.0).mean()
        print(f"  median views_burst_lift: {med_lift:.3f}")
        print(f"  share with lift > 1: {share_above_1:.1%} (n={len(lift_series)})")

# 4. KMeans
    cluster_cols = [c for c in FEATURES if c in features_pl.columns]
    print(f"\n=== Silhouette over k=2..6 (features: {cluster_cols}) ===")
    
    # 1. Strictly filter out nulls and NaNs across all cluster features in Polars
    clean_cluster_pl = features_pl.select(cluster_cols).filter(
        pl.all_horizontal(pl.col("*").is_not_null() & ~pl.col("*").is_nan())
    )
    
    X_mat = clean_cluster_pl.to_numpy()
    
    # 2. Scale features and drop any residual non-finite rows (e.g. constant feature zero-variance NaNs)
    scaled_mat = StandardScaler().fit_transform(X_mat)
    valid_mask = np.isfinite(scaled_mat).all(axis=1)
    scaled_mat = scaled_mat[valid_mask]

    sil_scores = {}
    for kk in K_RANGE:
        labs = KMeans(n_clusters=kk, random_state=RANDOM_STATE, n_init=10).fit_predict(scaled_mat)
        sil_scores[kk] = float(silhouette_score(scaled_mat, labs))
        print(f"  k={kk}: silhouette={sil_scores[kk]:.4f}")
        
    best_k = max(sil_scores, key=sil_scores.get) if k is None else k
    print(f"Using k={best_k}")

# Assign cluster labels
    cluster_labels = _cluster(features_pl, cluster_cols, best_k)
    
    # Cast float array with NaNs into Polars Int64 with nulls
    features_pl = features_pl.with_columns(
        pl.Series("cluster", cluster_labels).cast(pl.Int64, strict=False)
    )

# 5. Sensitivity Analysis
    snapshot = get_snapshot_date()
    videos_pd = load_videos()
    videos_pd["music_id"] = videos_pd["music_id"].astype(str)
    
    # Cast post_date explicitly to datetime64[ns] to prevent numpy resolution mismatch
    videos_pd["post_date"] = pd.to_datetime(videos_pd["post_date"]).dt.tz_localize(None).astype("datetime64[ns]")
    
    sens_corrs = _sensitivity_analysis(features_pl, videos_pd, snapshot)

    # 6. Age bias
    med_rho = float(features_pl["age_views_spearman"].median()) if "age_views_spearman" in features_pl.columns else np.nan
    print(f"\n=== Age bias (Spearman age vs views within audio) ===")
    print(f"median age_views_spearman: {med_rho:.3f}")

    # 7. Topic tables
    if "topic" in features_pl.columns:
        print("\n=== cluster x topic ===")
        ct_df = features_pl.select(["cluster", "topic"]).to_pandas()
        print(pd.crosstab(ct_df["cluster"], ct_df["topic"], margins=True).to_string())

    if "sentiment" in features_pl.columns:
        print("\n=== cluster x sentiment ===")
        sent_pd = features_pl.select(["cluster", "sentiment"]).to_pandas()
        sent = sent_pd["sentiment"]
        if pd.api.types.is_numeric_dtype(sent):
            bins_sent = pd.cut(sent, bins=[-np.inf, -0.2, 0.2, np.inf], labels=["neg", "neu", "pos"])
            print(pd.crosstab(sent_pd["cluster"], bins_sent, margins=True).to_string())
        else:
            print(pd.crosstab(sent_pd["cluster"], sent, margins=True).to_string())

    # Write output audio features via Polars
    features_pl.write_parquet(processed_path("audio_features.parquet"))

    meta = {
        "bimodal_views_centroid_u": bimodal_views,
        "median_age_views_spearman": med_rho,
        "kleinberg_s": KLEINBERG_S,
        "kleinberg_gamma": KLEINBERG_GAMMA,
        "sensitivity_correlations": sens_corrs,
        "cluster_features": cluster_cols,
        "silhouette_scores": {str(kk): v for kk, v in sil_scores.items()},
        "best_k": best_k,
    }
    with open(processed_path("lifespan_explore_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nCluster sizes:")
    cluster_counts = (
        features_pl.group_by("cluster")
        .len()
        .sort("cluster")
        .to_pandas()
        .set_index("cluster")
    )
    print(cluster_counts.to_string())
    print(f"runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    main(k=args.k)