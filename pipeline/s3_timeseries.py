"""Daily video counts per music_id (post-date) over the fixed analysis window."""
from __future__ import annotations

import time

import pandas as pd

from loader import get_snapshot_date, get_window, load_videos, processed_path


def main() -> None:
    t0 = time.perf_counter()
    videos = load_videos()
    window_start, window_end = get_window()
    snapshot = get_snapshot_date()

    videos = videos.copy()
    videos["post_date"] = pd.to_datetime(videos["post_date"]).dt.normalize()
    videos = videos[
        (videos["post_date"] >= window_start) & (videos["post_date"] <= window_end)
    ]

    daily = (
        videos.groupby(["music_id", "post_date"], as_index=False)
        .size()
        .rename(columns={"size": "video_count", "post_date": "date"})
    )

    # Expand to fixed window including zero days (per music_id)
    full_idx = pd.date_range(window_start, window_end, freq="D")
    rows = []
    for mid, grp in daily.groupby("music_id"):
        s = grp.set_index("date")["video_count"].reindex(full_idx, fill_value=0)
        part = s.rename("video_count").rename_axis("date").reset_index()
        part["music_id"] = mid
        rows.append(part)
    daily_posts = pd.concat(rows, ignore_index=True)
    daily_posts.to_parquet(processed_path("daily_posts.parquet"), index=False)

    # Same-day assumption: all of a video's total_views land on post_date
    same = (
        videos.groupby(["music_id", "post_date"], as_index=False)["total_views"]
        .sum()
        .rename(columns={"post_date": "date", "total_views": "views_same_day"})
    )
    same_rows = []
    for mid, grp in same.groupby("music_id"):
        s = grp.set_index("date")["views_same_day"].reindex(full_idx, fill_value=0)
        part = s.rename("views_same_day").rename_axis("date").reset_index()
        part["music_id"] = mid
        same_rows.append(part)
    same_day = pd.concat(same_rows, ignore_index=True)
    same_day.to_parquet(processed_path("daily_views_same_day.parquet"), index=False)

    elapsed = time.perf_counter() - t0
    print(f"daily_posts: {len(daily_posts):,} rows (zeros filled)")
    print(f"daily_views_same_day: {len(same_day):,} rows")
    print(f"window: {window_start.date()} -> {window_end.date()} (snapshot={snapshot.date()})")
    print(f"runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
