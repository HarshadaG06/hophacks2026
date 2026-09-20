"""Lifespan analysis: when within an audio's life are views concentrated?"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lifespan_common import (
    LIFESPAN_MODE,
    LIFESPAN_MODES,
    MIN_LIFESPAN_DAYS,
    MIN_VIDEOS,
    MIN_VIDEO_AGE_DAYS,
    N_BINS,
    TOP_VIDEO_QUANTILE,
    audio_bounds,
    lifespan_weeks,
)
from loader import get_snapshot_date, load_videos, processed_path


def _bin_index(u: np.ndarray, n_bins: int) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    idx = np.floor(u * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def _metrics_for_group(
    grp: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lifespan: int,
    snapshot: pd.Timestamp,
    *,
    weight_col: str,
    min_age_days: int = 0,
) -> dict | None:
    g = grp.copy()
    if min_age_days > 0:
        age = (snapshot - g["post_date"]).dt.days
        g = g[age >= min_age_days]
    if len(g) == 0:
        return None

    weights = g[weight_col].to_numpy(dtype=float)
    if weights.sum() <= 0:
        return None

    d = (g["post_date"] - start).dt.days.to_numpy(dtype=float)
    u = np.clip(d / max(lifespan, 1), 0.0, 1.0)

    views_centroid_u = float(np.average(u, weights=weights))
    count_centroid_u = float(u.mean())

    return {
        "views_centroid_u": views_centroid_u,
        "count_centroid_u": count_centroid_u,
        "centroid_gap": views_centroid_u - count_centroid_u,
        "share_views_first10": float(weights[u <= 0.10].sum() / weights.sum()),
        "share_views_first25": float(weights[u <= 0.25].sum() / weights.sum()),
        "share_views_middle50": float(weights[(u > 0.25) & (u <= 0.75)].sum() / weights.sum()),
        "share_views_last25": float(weights[u > 0.75].sum() / weights.sum()),
        "top_video_centroid_u": _top_quantile_centroid(u, weights, TOP_VIDEO_QUANTILE),
        "share_top_videos_in_first25": _share_top_in_first25(g, u, weights, TOP_VIDEO_QUANTILE),
        "top1_video_share_of_views": float(weights.max() / weights.sum()),
        "top1pct_share_of_views": _top1pct_share(weights),
        "age_views_spearman": _age_spearman(g, snapshot),
        "_u": u,
        "_weights": weights,
    }


def _top_quantile_centroid(u: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(u) == 0:
        return float("nan")
    thresh = np.quantile(weights, q)
    mask = weights >= thresh
    if not mask.any():
        return float("nan")
    return float(np.average(u[mask], weights=weights[mask]))


def _share_top_in_first25(
    g: pd.DataFrame, u: np.ndarray, weights: np.ndarray, q: float
) -> float:
    thresh = np.quantile(weights, q)
    top = weights >= thresh
    if not top.any():
        return float("nan")
    return float((u[top] <= 0.25).mean())


def _top1pct_share(weights: np.ndarray) -> float:
    n = len(weights)
    k = max(1, int(np.ceil(n * 0.01)))
    idx = np.argpartition(weights, -k)[-k:]
    return float(weights[idx].sum() / weights.sum())


def _age_spearman(g: pd.DataFrame, snapshot: pd.Timestamp) -> float:
    age = (snapshot - g["post_date"]).dt.days.to_numpy(dtype=float)
    views = g["total_views"].to_numpy(dtype=float)
    if len(age) < 3 or np.std(views) == 0:
        return float("nan")
    rho, _ = spearmanr(age, views)
    return float(rho)


def _daily_and_bins(
    grp: pd.DataFrame, start: pd.Timestamp, lifespan: int, music_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    # Ensure post_date is on or after start date
    valid_mask = grp["post_date"] >= start
    g_valid = grp[valid_mask] if not valid_mask.all() else grp

    d = (g_valid["post_date"] - start).dt.days.astype(int)
    views = g_valid["total_views"].to_numpy(dtype=float)

    # Clip days to ensure valid indexing within [0, lifespan]
    int_days = np.clip(d.to_numpy(dtype=int), 0, lifespan)

    # Fast bincount aggregation for daily series
    daily_views = np.bincount(int_days, weights=views, minlength=lifespan + 1)
    daily_counts = np.bincount(int_days, minlength=lifespan + 1)

    daily = pd.DataFrame({
        "music_id": [music_id] * (lifespan + 1),
        "day_index": np.arange(lifespan + 1, dtype=int),
        "views_sum": daily_views.astype(np.int64),
        "video_count": daily_counts.astype(int),
        "u": np.arange(lifespan + 1) / max(lifespan, 1),
    })

    u_vid = np.clip(int_days.astype(float) / max(lifespan, 1), 0.0, 1.0)
    bin_idx = _bin_index(u_vid, N_BINS)
    bin_views = np.bincount(bin_idx, weights=views, minlength=N_BINS)
    bin_counts = np.bincount(bin_idx, minlength=N_BINS)
    total_v = bin_views.sum()
    total_c = bin_counts.sum()

    bins = pd.DataFrame({
        "music_id": [music_id] * N_BINS,
        "bin": np.arange(N_BINS),
        "u_center": (np.arange(N_BINS) + 0.5) / N_BINS,
        "share_of_views": bin_views / total_v if total_v > 0 else 0.0,
        "share_of_videos": bin_counts / total_c if total_c > 0 else 0.0,
    })

    peak_day_views = int(daily.loc[daily["views_sum"].idxmax(), "day_index"]) if daily["views_sum"].sum() > 0 else 0
    peak_day_count = int(daily.loc[daily["video_count"].idxmax(), "day_index"]) if daily["video_count"].sum() > 0 else 0
    return daily, bins, peak_day_views, peak_day_count


def _weekly_from_daily(daily: pd.DataFrame, lifespan: int, music_id: str) -> pd.DataFrame:
    """Aggregate daily series to calendar weeks (week 0 = days 0–6 from audio_start)."""
    d = daily.copy()
    d["week_index"] = (d["day_index"] // 7).astype(int)
    weekly = d.groupby("week_index", as_index=False).agg(
        views_sum=("views_sum", "sum"),
        video_count=("video_count", "sum"),
    )
    n_weeks = lifespan_weeks(lifespan)
    full = pd.DataFrame({"week_index": np.arange(n_weeks, dtype=int)})
    weekly = full.merge(weekly, on="week_index", how="left").fillna(
        {"views_sum": 0, "video_count": 0}
    )
    weekly["music_id"] = music_id
    weekly["views_sum"] = weekly["views_sum"].astype(np.int64)
    weekly["video_count"] = weekly["video_count"].astype(int)
    weekly["u"] = (weekly["week_index"] * 7 + 3.5) / max(lifespan, 1)
    return weekly


def _process_mode(
    videos: pd.DataFrame, snapshot: pd.Timestamp, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_rows = []
    weekly_rows = []
    bin_rows = []
    audio_rows = []

    for music_id, grp in videos.groupby("music_id"):
        grp = grp.sort_values("post_date")
        if len(grp) < MIN_VIDEOS:
            continue
        start, end, lifespan = audio_bounds(grp["post_date"], mode, snapshot)
        if lifespan < MIN_LIFESPAN_DAYS:
            continue

        str_music_id = str(music_id)
        daily, bins, peak_d_v, peak_d_c = _daily_and_bins(grp, start, lifespan, str_music_id)
        daily_rows.append(daily)
        weekly_rows.append(_weekly_from_daily(daily, lifespan, str_music_id))
        bin_rows.append(bins)

        base = _metrics_for_group(
            grp, start, end, lifespan, snapshot, weight_col="total_views"
        )
        if base is None:
            continue
        old = _metrics_for_group(
            grp, start, end, lifespan, snapshot,
            weight_col="total_views", min_age_days=MIN_VIDEO_AGE_DAYS,
        )
        logm = _metrics_for_group(
            grp.assign(log_views=np.log1p(grp["total_views"])),
            start, end, lifespan, snapshot,
            weight_col="log_views",
        )

        row = {
            "music_id": str_music_id,
            "lifespan_days": lifespan,
            "n_videos": len(grp),
            "total_views": int(grp["total_views"].sum()),
            "views_centroid_u": base["views_centroid_u"],
            "count_centroid_u": base["count_centroid_u"],
            "centroid_gap": base["centroid_gap"],
            "peak_u_views": peak_d_v / max(lifespan, 1),
            "peak_u_count": peak_d_c / max(lifespan, 1),
            "share_views_first10": base["share_views_first10"],
            "share_views_first25": base["share_views_first25"],
            "share_views_middle50": base["share_views_middle50"],
            "share_views_last25": base["share_views_last25"],
            "top_video_centroid_u": base["top_video_centroid_u"],
            "share_top_videos_in_first25": base["share_top_videos_in_first25"],
            "top1_video_share_of_views": base["top1_video_share_of_views"],
            "top1pct_share_of_views": base["top1pct_share_of_views"],
            "age_views_spearman": base["age_views_spearman"],
        }
        for suffix, m in [("_old", old), ("_log", logm)]:
            if m is None:
                for k in (
                    "views_centroid_u", "count_centroid_u", "centroid_gap",
                    "share_views_first10", "share_views_first25",
                    "share_views_middle50", "share_views_last25",
                    "top_video_centroid_u",
                ):
                    row[f"{k}{suffix}"] = np.nan
            else:
                row[f"views_centroid_u{suffix}"] = m["views_centroid_u"]
                row[f"count_centroid_u{suffix}"] = m["count_centroid_u"]
                row[f"centroid_gap{suffix}"] = m["centroid_gap"]
                row[f"share_views_first10{suffix}"] = m["share_views_first10"]
                row[f"share_views_first25{suffix}"] = m["share_views_first25"]
                row[f"share_views_middle50{suffix}"] = m["share_views_middle50"]
                row[f"share_views_last25{suffix}"] = m["share_views_last25"]
                row[f"top_video_centroid_u{suffix}"] = m["top_video_centroid_u"]

        audio_rows.append(row)

    per_audio = pd.DataFrame(audio_rows)
    daily_all = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    weekly_all = pd.concat(weekly_rows, ignore_index=True) if weekly_rows else pd.DataFrame()
    bins_all = pd.concat(bin_rows, ignore_index=True) if bin_rows else pd.DataFrame()
    return per_audio, daily_all, weekly_all, bins_all


def main() -> None:
    t0 = time.perf_counter()
    snapshot = get_snapshot_date()
    videos = load_videos()
    videos["music_id"] = videos["music_id"].astype(str)
    videos["post_date"] = pd.to_datetime(videos["post_date"]).dt.normalize()

    print(f"LIFESPAN_MODE={LIFESPAN_MODE}  snapshot={snapshot.date()}")

    per_audio, daily, weekly, bins = _process_mode(videos, snapshot, LIFESPAN_MODE)
    per_audio.to_parquet(processed_path("lifespan_per_audio.parquet"), index=False)
    daily.to_parquet(processed_path("lifespan_daily.parquet"), index=False)
    weekly.to_parquet(processed_path("lifespan_weekly.parquet"), index=False)
    bins.to_parquet(processed_path("lifespan_bins.parquet"), index=False)

    # Robustness: median |delta views_centroid_u| across variants and modes
    print("\n=== Robustness (views_centroid_u) ===")
    base_c = per_audio["views_centroid_u"]
    for suffix in ("_old", "_log"):
        col = f"views_centroid_u{suffix}"
        if col in per_audio.columns:
            diff = (base_c - per_audio[col]).abs()
            print(f"  vs {suffix}: median |delta| = {diff.median():.4f}")

    centroids_by_mode = {LIFESPAN_MODE: base_c.reset_index(drop=True)}
    for mode in LIFESPAN_MODES:
        if mode == LIFESPAN_MODE:
            continue
        pa, _, _, _ = _process_mode(videos, snapshot, mode)
        merged = per_audio[["music_id", "views_centroid_u"]].merge(
            pa[["music_id", "views_centroid_u"]].rename(
                columns={"views_centroid_u": "views_centroid_u_alt"}
            ),
            on="music_id",
            how="inner",
        )
        if len(merged):
            diff = (merged["views_centroid_u"] - merged["views_centroid_u_alt"]).abs()
            print(f"  vs mode={mode}: median |delta| = {diff.median():.4f} (n={len(merged)})")

    print(f"\n=== Summary ===")
    print(f"audios kept: {len(per_audio):,}")
    print(f"median lifespan_days: {per_audio['lifespan_days'].median():.0f}")
    print(f"median views_centroid_u: {per_audio['views_centroid_u'].median():.3f}")
    print(f"median top_video_centroid_u: {per_audio['top_video_centroid_u'].median():.3f}")
    print(f"median age_views_spearman: {per_audio['age_views_spearman'].median():.3f}")
    print(f"runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()