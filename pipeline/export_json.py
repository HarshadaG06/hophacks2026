"""Export aggregated JSON for lifespan D3 viz."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lifespan_common import lifespan_weeks
from loader import load_processed, processed_path

VIZ_DATA = Path(__file__).resolve().parents[1] / "viz" / "data"
TOP_BURST_EXPORT_N = 50


def _mid(x) -> str:
    return str(x)


def _write(name: str, obj) -> None:
    path = VIZ_DATA / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), default=str)
    print(f"  {name}: {path.stat().st_size / 1024:.1f} KB")


def _hist_pack(values: pd.Series, bins: int = 20, x_range=None) -> dict:
    v = values.dropna().to_numpy()
    if len(v) == 0:
        return {"counts": [], "edges": []}
    counts, edges = np.histogram(v, bins=bins, range=x_range)
    return {
        "counts": counts.astype(int).tolist(),
        "edges": [round(float(e), 4) for e in edges],
    }


def _mean_curve(bins: pd.DataFrame, music_ids) -> list:
    sub = bins[bins["music_id"].isin(music_ids)]
    if sub.empty:
        return []
    agg = sub.groupby("bin", as_index=False).agg(
        share_of_views=("share_of_views", "mean"),
        share_of_videos=("share_of_videos", "mean"),
        u_center=("u_center", "first"),
    )
    return [
        {
            "bin": int(r.bin),
            "u": round(float(r.u_center), 4),
            "share_of_views": round(float(r.share_of_views), 6),
            "share_of_videos": round(float(r.share_of_videos), 6),
        }
        for r in agg.sort_values("bin").itertuples(index=False)
    ]


def main() -> None:
    t0 = time.perf_counter()
    VIZ_DATA.mkdir(parents=True, exist_ok=True)

    features = load_processed("audio_features.parquet")
    features["music_id"] = features["music_id"].map(_mid)
    bins = load_processed("lifespan_bins.parquet")
    bins["music_id"] = bins["music_id"].map(_mid)
    weekly = load_processed("lifespan_weekly.parquet")
    weekly["music_id"] = weekly["music_id"].map(_mid)

    topic_cols = [c for c in ("topic", "sentiment") if c in features.columns]
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

    audios = []
    for _, row in features.iterrows():
        rec = {"music_id": _mid(row["music_id"])}
        for c in metric_cols:
            if c not in row.index:
                continue
            v = row[c]
            if pd.isna(v):
                rec[c] = None
            elif c in ("lifespan_days", "n_videos", "total_views", "cluster", "kleinberg_n_bursts"):
                rec[c] = int(v)
            else:
                rec[c] = round(float(v), 4)
        for c in topic_cols:
            v = row[c]
            rec[c] = None if pd.isna(v) else (float(v) if c == "sentiment" else str(v))
        rec["lifespan_weeks"] = lifespan_weeks(int(row["lifespan_days"])) if pd.notna(row.get("lifespan_days")) else None
        audios.append(rec)
    _write("audios.json", audios)

    lifespan_bins = {}
    for mid, grp in bins.groupby("music_id"):
        lifespan_bins[_mid(mid)] = [
            {
                "bin": int(r.bin),
                "u": round(float(r.u_center), 4),
                "share_of_views": round(float(r.share_of_views), 6),
                "share_of_videos": round(float(r.share_of_videos), 6),
            }
            for r in grp.sort_values("bin").itertuples(index=False)
        ]
    _write("lifespan_bins.json", lifespan_bins)

    lifespan_weekly = {}
    active = weekly[(weekly["views_sum"] > 0) | (weekly["video_count"] > 0)]
    for mid, grp in active.groupby("music_id"):
        grp = grp.sort_values("week_index")
        lifespan_weekly[_mid(mid)] = {
            "views": [
                [int(r.week_index), int(r.views_sum)]
                for r in grp.itertuples(index=False)
                if r.views_sum > 0
            ],
            "counts": [
                [int(r.week_index), int(r.video_count)]
                for r in grp.itertuples(index=False)
                if r.video_count > 0
            ],
            "max_week": int(grp["week_index"].max()),
        }
    _write("lifespan_weekly.json", lifespan_weekly)

    features["lifespan_tercile"] = pd.qcut(
        features["lifespan_days"], q=3, labels=["short", "medium", "long"]
    )
    mean_curves = {
        "all": _mean_curve(bins, features["music_id"]),
        "short": _mean_curve(bins, features.loc[features["lifespan_tercile"] == "short", "music_id"]),
        "medium": _mean_curve(bins, features.loc[features["lifespan_tercile"] == "medium", "music_id"]),
        "long": _mean_curve(bins, features.loc[features["lifespan_tercile"] == "long", "music_id"]),
    }
    _write("lifespan_mean_curves.json", mean_curves)

    hist = _hist_pack(features["views_centroid_u"], bins=20, x_range=(0, 1))
    meta_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "lifespan_explore_meta.json"
    if meta_path.exists():
        hist.update(json.loads(meta_path.read_text(encoding="utf-8")))
    _write("centroid_hist.json", hist)

    burst_hist = {
        "kleinberg_max_level": _hist_pack(features["kleinberg_max_level"]),
    }
    _write("burstiness_hist.json", burst_hist)

    # Kleinberg intervals: top N by video count + all audios (for any selection)
    bursts = load_processed("kleinberg_bursts.parquet")
    bursts["music_id"] = bursts["music_id"].map(_mid)
    top_ids = set(
        features.nlargest(TOP_BURST_EXPORT_N, "n_videos")["music_id"].map(_mid).tolist()
    )
    export_ids = set(features["music_id"].map(_mid))
    kleinberg_bursts = {}
    for mid, grp in bursts.groupby("music_id"):
        mid = _mid(mid)
        if mid not in export_ids:
            continue
        kleinberg_bursts[mid] = [
            {
                "level": int(r.level),
                "start_week": round(float(getattr(r, "start_week", r.start_day / 7.0)), 4),
                "end_week": round(float(getattr(r, "end_week", r.end_day / 7.0)), 4),
                "start_u": round(float(r.start_u), 4),
                "end_u": round(float(r.end_u), 4),
            }
            for r in grp.sort_values(
                ["level", "start_week" if "start_week" in grp.columns else "start_day"],
                ascending=[False, True],
            ).itertuples(index=False)
            if int(r.level) >= 1
        ]
    _write("kleinberg_bursts.json", kleinberg_bursts)

    # L0 background scores (weekly Kleinberg units)
    l0 = bursts[bursts["level"] == 0].merge(
        features[["music_id", "n_videos", "lifespan_days"]], on="music_id", how="left"
    )
    l0_scores = {}
    for r in l0.itertuples(index=False):
        span_w = float(getattr(r, "end_week", r.end_day / 7.0) - getattr(r, "start_week", r.start_day / 7.0))
        n = int(r.n_videos) if pd.notna(r.n_videos) else None
        l0_scores[_mid(r.music_id)] = {
            "l0_span_weeks": round(span_w, 4),
            "l0_start_u": round(float(r.start_u), 6),
            "l0_end_u": round(float(r.end_u), 6),
            "l0_baseline_posts_per_week": round(n / span_w, 6) if n and span_w > 0 else None,
        }
    _write("kleinberg_l0_scores.json", l0_scores)
    (processed_path("kleinberg_l0_scores.json")).write_text(
        json.dumps(l0_scores, indent=2), encoding="utf-8"
    )

    print(f"Export complete. runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
