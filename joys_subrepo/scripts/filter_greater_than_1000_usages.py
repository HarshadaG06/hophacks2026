import polars as pl

# 1. Lazy scan the dataset without loading full file into memory
pl.scan_parquet("10M_unprocessed.parquet") \
    .select("music_id") \
    .group_by("music_id") \
    .agg(pl.len().alias("video_count")) \
    .filter(pl.col("video_count") > 1000) \
    .sink_parquet("greater_1000_usages.parquet")