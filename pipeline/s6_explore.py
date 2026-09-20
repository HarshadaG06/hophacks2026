"""Unsupervised exploration: lifespan view concentration + burstiness."""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed

from loader import PROCESSED, RAW, get_snapshot_date, load_processed, load_videos, processed_path
from s5_burstiness import KLEINBERG_GAMMA, KLEINBERG_S, kleinberg_sensitivity_row

# Tunables
K_RANGE = range(2, 7)
RANDOM_STATE = 42
SENSITIVITY_N = 100
SENSITIVITY_GAMMAS = (0.5, 1.0, 2.0)
SENSITIVITY_S = (2, 3)
# Cap events for sensitivity reruns only — keeps 600 Kleinberg calls tractable.
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


def _cluster(features: pd.DataFrame, cols: list[str], k: int) -> pd.Series:
    X = features[cols].copy()
    mask = X.notna().all(axis=1)
    labels = pd.Series(np.nan, index=features.index, dtype="float")
    if mask.sum() < k:
        return labels
    scaled = StandardScaler().fit_transform(X.loc[mask])
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels.loc[mask] = km.fit_predict(scaled)
    return labels


def _print_histogram(name: str, values: pd.Series, bins: int = 20, x_range=None) -> None:
    v = values.dropna().to_numpy()
    if len(v) == 0:
        print(f"{name}: no data")
        return
    counts, edges = np.histogram(v, bins=bins, range=x_range)
    print(f"{name}: n={len(v)}  hist={counts.tolist()}")


def _sensitivity_analysis(
    per_audio: pd.DataFrame,
    videos: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    sample_ids = rng.choice(
        per_audio["music_id"].tolist(),
        size=min(SENSITIVITY_N, len(per_audio)),
        replace=False,
    )
    groups = {mid: g for mid, g in videos.groupby("music_id") if mid in set(sample_ids)}

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

    sens = pd.DataFrame(setting_rows)
    if sens.empty:
        return {}

    # Spearman correlation of each metric across parameter settings (pivot wide)
    corrs = {}
    for col in ("kleinberg_max_level", "share_of_views_in_bursts"):
        wide = sens.pivot_table(index="music_id", columns=["s", "gamma"], values=col)
        vals = wide.to_numpy().ravel()
        # pairwise Spearman across settings for same audio — compare rank stability
        rhos = []
        cols = list(wide.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = wide.iloc[:, i].to_numpy()
                b = wide.iloc[:, j].to_numpy()
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
    features = load_processed("lifespan_per_audio.parquet")
    features["music_id"] = features["music_id"].astype(str)
    features["log_lifespan_days"] = np.log1p(features["lifespan_days"])

    topics_path = RAW / "topics.parquet"
    if not topics_path.exists():
        topics_path = PROCESSED / "topics.parquet"
    if topics_path.exists():
        topics = pd.read_parquet(topics_path)
        topics["music_id"] = topics["music_id"].astype(str)
        features = features.merge(topics, on="music_id", how="left")
        print(f"Merged topics from {topics_path}")
    else:
        print("No topics.parquet found — skipping topic/sentiment merge")

    # 1. Histograms + GMM
    print("\n=== Kleinberg histogram ===")
    _print_histogram("kleinberg_max_level", features["kleinberg_max_level"])

    print("\n=== Bimodality (GMM BIC) ===")
    bimodal_views = _gmm_bic("views_centroid_u", features["views_centroid_u"].to_numpy(), (0, 1))

    # 2. Relationship checks
    print("\n=== views_centroid_u vs burst_views_centroid_u ===")
    sub = features[["views_centroid_u", "burst_views_centroid_u"]].dropna()
    if len(sub) >= 5:
        rho, p = spearmanr(sub["views_centroid_u"], sub["burst_views_centroid_u"])
        print(f"  Spearman rho={rho:.3f}  p={p:.2e}  n={len(sub)}")
        print(sub.describe().to_string())

    print("\n=== Spearman: burst features vs lifespan features ===")
    corr_cols = [c for c in BURST_CORR_COLS + LIFESPAN_CORR_COLS if c in features.columns]
    corr_df = features[corr_cols].corr(method="spearman")
    burst_part = corr_df.loc[
        [c for c in BURST_CORR_COLS if c in corr_df.index],
        [c for c in LIFESPAN_CORR_COLS if c in corr_df.columns],
    ]
    print(burst_part.round(3).to_string())

    # 3. Burst membership vs concentration
    print("\n=== Does burst membership explain concentration? ===")
    lift = features["views_burst_lift"].dropna()
    if len(lift):
        print(f"  median views_burst_lift: {lift.median():.3f}")
        print(f"  share with lift > 1: {(lift > 1).mean():.1%} (n={len(lift)})")

    # 4. KMeans
    cluster_cols = [c for c in FEATURES if c in features.columns]
    print(f"\n=== Silhouette over k=2..6 (features: {cluster_cols}) ===")
    X = features[cluster_cols].dropna()
    scaled = StandardScaler().fit_transform(X)
    sil_scores = {}
    for kk in K_RANGE:
        labs = KMeans(n_clusters=kk, random_state=RANDOM_STATE, n_init=10).fit_predict(scaled)
        sil_scores[kk] = float(silhouette_score(scaled, labs))
        print(f"  k={kk}: silhouette={sil_scores[kk]:.4f}")
    best_k = max(sil_scores, key=sil_scores.get) if k is None else k
    print(f"Using k={best_k}")

    features["cluster"] = _cluster(features, cluster_cols, best_k).astype("Int64")

    # 5. Sensitivity
    snapshot = get_snapshot_date()
    videos = load_videos()
    videos["music_id"] = videos["music_id"].astype(str)
    videos["post_date"] = pd.to_datetime(videos["post_date"]).dt.normalize()
    sens_corrs = _sensitivity_analysis(features, videos, snapshot)

    # 6. Age bias
    med_rho = float(features["age_views_spearman"].median())
    print(f"\n=== Age bias (Spearman age vs views within audio) ===")
    print(f"median age_views_spearman: {med_rho:.3f}")

    # 7. Topic tables
    if "topic" in features.columns:
        print("\n=== cluster x topic ===")
        print(pd.crosstab(features["cluster"], features["topic"], margins=True).to_string())
    if "sentiment" in features.columns:
        print("\n=== cluster x sentiment ===")
        sent = features["sentiment"]
        if pd.api.types.is_numeric_dtype(sent):
            bins_sent = pd.cut(sent, bins=[-np.inf, -0.2, 0.2, np.inf], labels=["neg", "neu", "pos"])
            print(pd.crosstab(features["cluster"], bins_sent, margins=True).to_string())
        else:
            print(pd.crosstab(features["cluster"], sent, margins=True).to_string())

    features.to_parquet(processed_path("audio_features.parquet"), index=False)

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
    print(features["cluster"].value_counts(dropna=False).sort_index().to_string())
    print(f"runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    main(k=args.k)
