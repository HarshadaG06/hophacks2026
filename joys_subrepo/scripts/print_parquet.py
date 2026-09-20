import polars as pl

# Fetch and display first 100 rows
df = pl.read_parquet("greater_1000_output_videos.parquet", n_rows=100)

# Show all columns without truncation
with pl.Config(tbl_cols=-1, tbl_rows=100):
    print(df)