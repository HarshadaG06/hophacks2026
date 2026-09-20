import polars as pl

# 1. Scan the full dataset lazily
full_dataset = pl.scan_parquet("10M_unprocessed.parquet")

# 2. Scan the shortlist lazily
shortlist = pl.scan_parquet("greater_1000_usages.parquet").select("music_id")

# 3. Inner join to filter videos and stream directly to disk
full_dataset \
    .join(shortlist, on="music_id", how="inner") \
    .sink_parquet("greater_1000_output_videos.parquet")