"""Daily video counts per music_id (post-date) over the fixed analysis window using Polars."""
from __future__ import annotations

import time
import polars as pl

from loader import get_snapshot_date, get_window, processed_path, REAL_PATH


def main() -> None:
    t0 = time.perf_counter()
    window_start, window_end = get_window()
    snapshot = get_snapshot_date()

    # 1. Lazy evaluation directly from parquet (avoids loading unnecessary data into memory)
    lazy_df = (
        pl.scan_parquet(REAL_PATH)
        .with_columns([
            # Convert Unix timestamps to Date objects directly
            pl.from_epoch(pl.col("create_time"), time_unit="s").dt.date().alias("date"),
            pl.col("music_id").round().cast(pl.Int64).cast(pl.String).alias("music_id"),
            pl.col("play_count").cast(pl.Int64).fill_null(0).alias("total_views"),
        ])
        .filter(
            (pl.col("date") >= window_start.date()) & 
            (pl.col("date") <= window_end.date())
        )
    )

    # 2. Parallelized GroupBy aggregation
    aggregated = (
        lazy_df.group_by(["music_id", "date"])
        .agg([
            pl.len().alias("video_count"),
            pl.col("total_views").sum().alias("views_same_day"),
        ])
        .collect()  # Triggers parallel processing engine
    )

    # 3. Fast grid expansion using cross join (music_ids x full date range)
    unique_musics = aggregated.select("music_id").unique()
    date_range = pl.date_range(
        start=window_start.date(),
        end=window_end.date(),
        interval="1d",
        eager=True,
    ).alias("date")

    # Cartesian product grid
    grid = unique_musics.join(date_range.to_frame(), how="cross")

    # 4. Join aggregated data onto grid and fill missing dates with 0
    full_df = (
        grid.join(aggregated, on=["music_id", "date"], how="left")
        .with_columns([
            pl.col("video_count").fill_null(0),
            pl.col("views_same_day").fill_null(0),
        ])
    )

    # 5. Write outputs in parallel
    full_df.select(["music_id", "date", "video_count"]).write_parquet(
        processed_path("daily_posts.parquet")
    )
    full_df.select(["music_id", "date", "views_same_day"]).write_parquet(
        processed_path("daily_views_same_day.parquet")
    )

    elapsed = time.perf_counter() - t0
    print(f"daily_posts: {len(full_df):,} rows (zeros filled)")
    print(f"window: {window_start.date()} -> {window_end.date()} (snapshot={snapshot.date()})")
    print(f"runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()