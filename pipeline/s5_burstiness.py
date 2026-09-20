"""
Kleinberg burst detection on each audio's own lifespan (weekly resolution).

Optimized with Polars for fast grouped expressions, zero-copy memory operations,
and fast Parquet IO.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed

from kleinberg import kleinberg
from lifespan_common import (
    LIFESPAN_MODE,
    TOP_VIDEO_QUANTILE,
    audio_bounds,
    lifespan_weeks,
    week_offset_to_u,
)
from loader import get_snapshot_date, load_processed, load_videos, processed_path

# Tunables
KLEINBERG_S = 2.0
KLEINBERG_GAMMA = 1.0
MIN_EVENTS = 20
MAX_EVENTS = None  # cap only if runtime demands it (random subsample, fixed seed)
JITTER_SEED = 0
BURST_LEVEL_MIN = 1
N_JOBS = -1


def _jittered_week_offsets(day_indices: np.ndarray, music_id: str) -> np.ndarray:
    """Day indices -> week offsets with seeded jitter inside each week."""
    rng = np.random.default_rng(JITTER_SEED ^ (hash(str(music_id)) & 0xFFFFFFFF))
    weeks = np.floor(day_indices.astype(float) / 7.0)
    return weeks + rng.uniform(0.0, 0.999, size=len(day_indices))


def _maybe_subsample(
    offsets: np.ndarray, music_id: str, max_events: int | None = None
) -> np.ndarray:
    cap = MAX_EVENTS if max_events is None else max_events
    if cap is None or len(offsets) <= cap:
        return offsets
    rng = np.random.default_rng(JITTER_SEED ^ (hash(str(music_id)) & 0xFFFFFFFF))
    keep = np.concatenate(
        [
            [0, len(offsets) - 1],
            rng.choice(np.arange(1, len(offsets) - 1), size=cap - 2, replace=False),
        ]
    )
    return np.sort(offsets[keep])


def _in_any_burst(week_offset: float, intervals: list[tuple[float, float, int]]) -> bool:
    for start, end, lvl in intervals:
        if lvl >= BURST_LEVEL_MIN and start <= week_offset <= end:
            return True
    return False


def compute_kleinberg_for_audio(
    music_id: str,
    days: np.ndarray,
    views: np.ndarray,
    lifespan: int,
    *,
    s: float = KLEINBERG_S,
    gamma: float = KLEINBERG_GAMMA,
    max_events: int | None = None,
) -> tuple[dict, list[dict]]:
    """Return (feature dict, interval row dicts)."""
    nan_feats = {
        "kleinberg_max_level": np.nan,
        "kleinberg_n_bursts": np.nan,
        "top_burst_start_u": np.nan,
        "top_burst_end_u": np.nan,
        "top_burst_length_share": np.nan,
        "share_of_videos_in_bursts": np.nan,
        "share_of_views_in_bursts": np.nan,
        "share_of_views_in_top_burst": np.nan,
        "top_video_in_burst_share": np.nan,
        "burst_views_centroid_u": np.nan,
        "views_burst_lift": np.nan,
    }
    n = len(days)
    if n < MIN_EVENTS:
        return nan_feats, []

    week_offsets_event = days / 7.0
    n_weeks = lifespan_weeks(lifespan)
    offsets = np.sort(
        _maybe_subsample(_jittered_week_offsets(days, music_id), music_id, max_events=max_events)
    )

    try:
        bursts = kleinberg(offsets, s=s, gamma=gamma)
    except Exception:
        return nan_feats, []

    if bursts is None or len(bursts) == 0:
        return nan_feats, []

    levels = bursts[:, 0].astype(int)
    starts = bursts[:, 1].astype(float)
    ends = bursts[:, 2].astype(float)

    interval_rows = [
        {
            "music_id": music_id,
            "level": int(lvl),
            "start_week": float(st),
            "end_week": float(en),
            "start_day": float(st * 7.0),
            "end_day": float(en * 7.0),
            "start_u": week_offset_to_u(st, lifespan),
            "end_u": week_offset_to_u(en, lifespan),
        }
        for lvl, st, en in zip(levels, starts, ends)
    ]

    burst_intervals = [
        (st, en, lv) for st, en, lv in zip(starts, ends, levels) if lv >= BURST_LEVEL_MIN
    ]

    u = np.clip(days / max(lifespan, 1), 0.0, 1.0)
    in_burst = np.array([_in_any_burst(w, burst_intervals) for w in week_offsets_event])
    share_videos = float(in_burst.mean()) if n else np.nan
    share_views = float(views[in_burst].sum() / views.sum()) if views.sum() > 0 else np.nan

    sub = [(st, en, lv) for st, en, lv in zip(starts, ends, levels) if lv >= BURST_LEVEL_MIN]
    top_start_u = top_end_u = top_length_share = share_views_top = np.nan
    if sub:
        sub.sort(key=lambda x: (-x[2], -(x[1] - x[0])))
        tst, ten, _ = sub[0]
        top_start_u = week_offset_to_u(tst, lifespan)
        top_end_u = week_offset_to_u(ten, lifespan)
        top_length_share = float((ten - tst) / max(n_weeks, 1))
        in_top = (week_offsets_event >= tst) & (week_offsets_event <= ten)
        share_views_top = float(views[in_top].sum() / views.sum()) if views.sum() > 0 else np.nan

    thresh = np.quantile(views, TOP_VIDEO_QUANTILE) if n > 1 else views.max()
    top_mask = views >= thresh
    top_in_burst = float((in_burst & top_mask).sum() / top_mask.sum()) if top_mask.any() else np.nan

    burst_centroid = (
        float(np.average(u[in_burst], weights=views[in_burst]))
        if in_burst.any() and views[in_burst].sum() > 0
        else np.nan
    )
    lift = float(share_views / share_videos) if share_videos and share_videos > 0 else np.nan

    feats = {
        "kleinberg_max_level": float(levels.max()),
        "kleinberg_n_bursts": int(sum(1 for lv in levels if lv >= BURST_LEVEL_MIN)),
        "top_burst_start_u": top_start_u,
        "top_burst_end_u": top_end_u,
        "top_burst_length_share": top_length_share,
        "share_of_videos_in_bursts": share_videos,
        "share_of_views_in_bursts": share_views,
        "share_of_views_in_top_burst": share_views_top,
        "top_video_in_burst_share": top_in_burst,
        "burst_views_centroid_u": burst_centroid,
        "views_burst_lift": lift,
    }
    return feats, interval_rows


def _process_one(
    music_id: str,
    post_dates: np.ndarray,
    views: np.ndarray,
    snapshot: pl.Timestamp,
    s: float,
    gamma: float,
    max_events: int | None = None,
) -> tuple[dict, list[dict]]:
    # Convert numpy dates to pandas Timestamp series for compatibility with audio_bounds
    pd_post_dates = pl.Series(post_dates).to_pandas()
    start, _, lifespan = audio_bounds(pd_post_dates, LIFESPAN_MODE, snapshot)
    
    # Calculate day offsets relative to audio_start
    days = (pd_post_dates - start).dt.days.to_numpy(dtype=float)
    days = np.clip(days, 0, None)  # Ensure non-negative day offsets

    kfeats, intervals = compute_kleinberg_for_audio(
        music_id, days, views, lifespan, s=s, gamma=gamma, max_events=max_events
    )
    return kfeats, intervals


def main() -> None:
    t0 = time.perf_counter()
    snapshot = get_snapshot_date()

    # Load videos via Pandas/loader then convert to Polars for high-speed grouping
    videos_pd = load_videos()
    videos_pd["music_id"] = videos_pd["music_id"].astype(str)
    videos_pd["post_date"] = pd.to_datetime(videos_pd["post_date"]).dt.normalize()
    
    videos_pl = pl.from_pandas(videos_pd)

    # Load processed lifespan_per_audio via Polars
    per_audio_pl = pl.read_parquet(processed_path("lifespan_per_audio.parquet"))
    per_audio_pl = per_audio_pl.with_columns(pl.col("music_id").cast(pl.Utf8))

    eligible_ids = set(per_audio_pl["music_id"].to_list())
    
    # Filter videos to eligible music_ids
    filtered_videos = videos_pl.filter(pl.col("music_id").is_in(eligible_ids))

    # Group by music_id and extract numpy arrays for joblib execution
    grouped_data = {}
    for group in filtered_videos.group_by("music_id", maintain_order=False):
        mid = group[0][0]
        gdf = group[1].sort("post_date")
        grouped_data[mid] = (
            gdf["post_date"].to_numpy(),
            gdf["total_views"].cast(pl.Float64).to_numpy(),
        )

    # Benchmark single audio performance
    sample_mid = per_audio_pl.sort("n_videos", descending=True)["music_id"][0]
    sample_dates, sample_views = grouped_data[sample_mid]
    
    t_one = time.perf_counter()
    _process_one(sample_mid, sample_dates, sample_views, snapshot, KLEINBERG_S, KLEINBERG_GAMMA)
    print(
        f"Timing: music_id={sample_mid} n={len(sample_dates):,} "
        f"took {time.perf_counter() - t_one:.2f}s (MAX_EVENTS={MAX_EVENTS})"
    )

    print(f"Kleinberg (weekly) s={KLEINBERG_S} gamma={KLEINBERG_GAMMA} on {len(grouped_data)} audios ...")
    t_k = time.perf_counter()
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_process_one)(mid, dates, views, snapshot, KLEINBERG_S, KLEINBERG_GAMMA)
        for mid, (dates, views) in grouped_data.items()
    )
    print(f"  wall time: {time.perf_counter() - t_k:.1f}s")

    feat_rows, all_intervals = zip(*results) if results else ([], [])
    
    # Construct Polars DataFrames from results
    burst_df = pl.DataFrame(feat_rows)
    burst_df = burst_df.with_columns(pl.Series("music_id", list(grouped_data.keys()), dtype=pl.Utf8))
    
    intervals_flat = [r for sub in all_intervals for r in sub]
    intervals_df = pl.DataFrame(intervals_flat) if intervals_flat else pl.DataFrame()

    # Drop existing burst columns if present to avoid duplication on join
    burst_cols = [c for c in burst_df.columns if c != "music_id"]
    legacy_cols = ["burstiness_count", "burstiness_views", "burstiness_views_log"]
    drop_cols = [c for c in (*burst_cols, *legacy_cols) if c in per_audio_pl.columns]
    
    out_pl = per_audio_pl.drop(drop_cols).join(burst_df, on="music_id", how="left")

    # Save output parquet files using Polars
    out_pl.write_parquet(processed_path("lifespan_per_audio.parquet"))
    intervals_df.write_parquet(processed_path("kleinberg_bursts.parquet"))

    has_burst = out_pl["kleinberg_n_bursts"].fill_null(0) > 0
    print("\n=== Summary ===")
    print(f"audios with bursts (level>={BURST_LEVEL_MIN}): {int(has_burst.sum()):,} / {len(out_pl):,}")
    print(f"median kleinberg_max_level: {out_pl['kleinberg_max_level'].median():.2f}")
    print(f"median views_burst_lift: {out_pl['views_burst_lift'].median():.3f}")
    print(f"kleinberg_bursts intervals: {len(intervals_df):,}")
    print(f"runtime: {time.perf_counter() - t0:.1f}s")

def kleinberg_sensitivity_row(
    music_id: str,
    group_df: pd.DataFrame | pl.DataFrame,
    snapshot: pd.Timestamp,
    s: float,
    gamma: float,
    max_events: int | None = None,
) -> dict:
    """Helper function for sensitivity analysis in downstream exploration scripts."""
    if isinstance(group_df, pl.DataFrame):
        post_dates = group_df["post_date"].cast(pl.Datetime("ns")).to_numpy()
        views = group_df["total_views"].cast(pl.Float64).to_numpy()
    else:
        post_dates = pd.to_datetime(group_df["post_date"]).to_numpy(dtype="datetime64[ns]")
        views = group_df["total_views"].to_numpy(dtype=float)

    feats, _ = _process_one(
        music_id=music_id,
        post_dates=post_dates,
        views=views,
        snapshot=snapshot,
        s=s,
        gamma=gamma,
        max_events=max_events,
    )
    
    row = {"music_id": music_id, "s": s, "gamma": gamma}
    row.update(feats)
    return row

if __name__ == "__main__":
    main()