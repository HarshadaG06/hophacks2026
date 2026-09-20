"""Export aggregated JSON for lifespan D3 viz.

Optimized with Polars for high-performance tabular aggregations,
nested list transformations, and fast JSON serialization.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl

from lifespan_common import lifespan_weeks
from loader import load_processed, processed_path

VIZ_DATA = Path(__file__).resolve().parents[1] / "viz" / "data"
TOP_BURST_EXPORT_N = 50


def _json_safe(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _write(name: str, obj) -> None:
    path = VIZ_DATA / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, separators=(",", ":"), default=str, allow_nan=False)
    print(f"  {name}: {path.stat().st_size / 1024:.1f} KB")


def _hist_pack(series: pl.Series, bins: int = 20, x_range=None) -> dict:
    v = series.drop_nulls().to_numpy()
    if len(v) == 0:
        return {"counts": [], "edges": []}
    counts, edges = np.histogram(v, bins=bins, range=x_range)
    return {
        "counts": counts.astype(int).tolist(),
        "edges": [round(float(e), 4) for e in edges],
    }


def _mean_curve(bins_pl: pl.DataFrame, music_ids: list[str]) -> list[dict]:
    sub = bins_pl.filter(pl.col("music_id").is_in(music_ids))
    if sub.is_empty():
        return []
    
    agg = (
        sub.group_by("bin")
        .agg(
            pl.col("share_of_views").mean(),
            pl.col("share_of_videos").mean(),
            pl.col("u_center").first(),
        )
        .sort("bin")
    )
    
    return [
        {
            "bin": int(r["bin"]),
            "u": round(float(r["u_center"]), 4),
            "share_of_views": round(float(r["share_of_views"]), 6),
            "share_of_videos": round(float(r["share_of_videos"]), 6),
        }
        for r in agg.to_dicts()
    ]


def main() -> None:
    t0 = time.perf_counter()
    VIZ_DATA.mkdir(parents=True, exist_ok=True)

    # 1. Load Parquet files with Polars
    features_pl = pl.read_parquet(processed_path("audio_features.parquet")).with_columns(
        pl.col("music_id").cast(pl.Utf8)
    )
    bins_pl = pl.read_parquet(processed_path("lifespan_bins.parquet")).with_columns(
        pl.col("music_id").cast(pl.Utf8)
    )
    weekly_pl = pl.read_parquet(processed_path("lifespan_weekly.parquet")).with_columns(
        pl.col("music_id").cast(pl.Utf8)
    )

    topic_cols = [c for c in ("topic", "sentiment") if c in features_pl.columns]
    metric_cols = [
        "lifespan_days", "n_videos", "total_views",
        "views_centroid_u", "count_centroid_u", "centroid_gap",
        "peak_u_views", "peak_u_count",
        "share_views_first10", "share_views_first25",
        "share_views_middle50", "share_views_last25",
        "top_video_centroid_u", "share_top_videos_in_first25",
        "top1_video_share_of_views", "top1pct_share_of_views",
        "age_views_spearman", "cluster",
        "kleinberg_max_level", "kleinberg_n_bursts",
        "top_burst_start_u", "top_burst_end_u", "top_burst_length_share",
        "share_of_videos_in_bursts", "share_of_views_in_bursts",
        "share_of_views_in_top_burst", "top_video_in_burst_share",
        "burst_views_centroid_u", "views_burst_lift",
    ]

    int_cols = {"lifespan_days", "n_videos", "total_views", "cluster", "kleinberg_n_bursts"}
    
    # Pre-process columns in Polars for clean JSON serializability
    select_exprs = [pl.col("music_id")]
    for c in metric_cols:
        if c in features_pl.columns:
            if c in int_cols:
                select_exprs.append(pl.col(c).cast(pl.Int64, strict=False))
            else:
                select_exprs.append(pl.col(c).cast(pl.Float64).round(4))

    for c in topic_cols:
        if c in features_pl.columns:
            if c == "sentiment":
                select_exprs.append(pl.col(c).cast(pl.Float64))
            else:
                select_exprs.append(pl.col(c).cast(pl.Utf8))

    export_features = features_pl.select(select_exprs)
    
    # Calculate lifespan_weeks for export
    audios = export_features.to_dicts()
    for rec in audios:
        days = rec.get("lifespan_days")
        rec["lifespan_weeks"] = lifespan_weeks(int(days)) if days is not None else None

    _write("audios.json", audios)

    # 2. Lifespan Bins JSON
    bins_grouped = (
        bins_pl.sort(["music_id", "bin"])
        .select([
            pl.col("music_id"),
            pl.struct([
                pl.col("bin").cast(pl.Int64),
                pl.col("u_center").round(4).alias("u"),
                pl.col("share_of_views").round(6),
                pl.col("share_of_videos").round(6),
            ]).alias("bin_data")
        ])
        .group_by("music_id")
        .agg(pl.col("bin_data"))
    )
    lifespan_bins = {r["music_id"]: r["bin_data"] for r in bins_grouped.to_dicts()}
    _write("lifespan_bins.json", lifespan_bins)

    # 3. Lifespan Weekly JSON
    active_weekly = weekly_pl.filter(
        (pl.col("views_sum") > 0) | (pl.col("video_count") > 0)
    ).sort(["music_id", "week_index"])

    weekly_grouped = (
        active_weekly.group_by("music_id")
        .agg(
            pl.col("week_index").max().cast(pl.Int64).alias("max_week"),
            pl.struct([pl.col("week_index").cast(pl.Int64), pl.col("views_sum").cast(pl.Int64)])
            .filter(pl.col("views_sum") > 0)
            .alias("views_pairs"),
            pl.struct([pl.col("week_index").cast(pl.Int64), pl.col("video_count").cast(pl.Int64)])
            .filter(pl.col("video_count") > 0)
            .alias("counts_pairs"),
        )
    )

    lifespan_weekly = {}
    for r in weekly_grouped.to_dicts():
        lifespan_weekly[r["music_id"]] = {
            "views": [[p["week_index"], p["views_sum"]] for p in r["views_pairs"]],
            "counts": [[p["week_index"], p["video_count"]] for p in r["counts_pairs"]],
            "max_week": r["max_week"],
        }
    _write("lifespan_weekly.json", lifespan_weekly)

    # 4. Mean Curves (Lifespan Terciles via Polars)
    terciles = features_pl.select([
        pl.col("music_id"),
        pl.col("lifespan_days").qcut(3, labels=["short", "medium", "long"]).alias("tercile")
    ])

    all_ids = features_pl["music_id"].to_list()
    short_ids = terciles.filter(pl.col("tercile") == "short")["music_id"].to_list()
    medium_ids = terciles.filter(pl.col("tercile") == "medium")["music_id"].to_list()
    long_ids = terciles.filter(pl.col("tercile") == "long")["music_id"].to_list()

    mean_curves = {
        "all": _mean_curve(bins_pl, all_ids),
        "short": _mean_curve(bins_pl, short_ids),
        "medium": _mean_curve(bins_pl, medium_ids),
        "long": _mean_curve(bins_pl, long_ids),
    }
    _write("lifespan_mean_curves.json", mean_curves)

    # 5. Histograms & Metadata
    hist = _hist_pack(features_pl["views_centroid_u"], bins=20, x_range=(0, 1))
    meta_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "lifespan_explore_meta.json"
    if meta_path.exists():
        hist.update(json.loads(meta_path.read_text(encoding="utf-8")))
    _write("centroid_hist.json", hist)

    burst_hist = {
        "kleinberg_max_level": _hist_pack(features_pl["kleinberg_max_level"]),
    }
    _write("burstiness_hist.json", burst_hist)

    # 6. Kleinberg Bursts JSON
    bursts_pl = pl.read_parquet(processed_path("kleinberg_bursts.parquet")).with_columns(
        pl.col("music_id").cast(pl.Utf8)
    )
    
    # Calculate start/end week fallback if not present
    if "start_week" not in bursts_pl.columns:
        bursts_pl = bursts_pl.with_columns(
            (pl.col("start_day") / 7.0).alias("start_week"),
            (pl.col("end_day") / 7.0).alias("end_week"),
        )

    export_ids = set(features_pl["music_id"].to_list())
    filtered_bursts = bursts_pl.filter(
        (pl.col("music_id").is_in(export_ids)) & (pl.col("level") >= 1)
    )

    bursts_grouped = (
        filtered_bursts.sort(["level", "start_week"], descending=[True, False])
        .select([
            pl.col("music_id"),
            pl.struct([
                pl.col("level").cast(pl.Int64),
                pl.col("start_week").round(4),
                pl.col("end_week").round(4),
                pl.col("start_u").round(4),
                pl.col("end_u").round(4),
            ]).alias("burst_data")
        ])
        .group_by("music_id")
        .agg(pl.col("burst_data"))
    )
    
    kleinberg_bursts = {r["music_id"]: r["burst_data"] for r in bursts_grouped.to_dicts()}
    _write("kleinberg_bursts.json", kleinberg_bursts)

    # 7. L0 Baseline Scores
    l0_pl = (
        bursts_pl.filter(pl.col("level") == 0)
        .join(
            features_pl.select(["music_id", "n_videos", "lifespan_days"]),
            on="music_id",
            how="left",
        )
        .with_columns(
            (pl.col("end_week") - pl.col("start_week")).alias("span_w")
        )
        .with_columns(
            pl.when(pl.col("span_w") > 0)
            .then(pl.col("n_videos") / pl.col("span_w"))
            .otherwise(None)
            .alias("l0_baseline_posts_per_week")
        )
    )

    l0_scores = {}
    for r in l0_pl.to_dicts():
        mid = r["music_id"]
        span_w = r["span_w"]
        l0_scores[mid] = {
            "l0_span_weeks": round(float(span_w), 4) if span_w is not None else None,
            "l0_start_u": round(float(r["start_u"]), 6) if r["start_u"] is not None else None,
            "l0_end_u": round(float(r["end_u"]), 6) if r["end_u"] is not None else None,
            "l0_baseline_posts_per_week": (
                round(float(r["l0_baseline_posts_per_week"]), 6)
                if r["l0_baseline_posts_per_week"] is not None
                else None
            ),
        }

    _write("kleinberg_l0_scores.json", l0_scores)
    (processed_path("kleinberg_l0_scores.json")).write_text(
        json.dumps(l0_scores, indent=2), encoding="utf-8"
    )

    print(f"Export complete. runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
