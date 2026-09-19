## Clean a large Parquet file

Install the dependency:

```powershell
python -m pip install .
```

Run the cleaner:

```powershell
python clean_parquet.py input.parquet cleaned.parquet
```

The script uses Polars' lazy streaming engine, so it does not load the entire
file into memory. It removes rows in this order:

1. Null or blank `music_id`
2. Null, blank, or invalid `create_time`
3. `is_ad = 1`

It writes the remaining rows to `cleaned.parquet` and writes counts and
percentages to `cleaned.parquet.cleanup_report.json`. The reason categories
are mutually exclusive, so they add up to the total number of deleted rows.

For string timestamps with a known format:

```powershell
python clean_parquet.py input.parquet cleaned.parquet `
  --timestamp-format "%Y-%m-%d %H:%M:%S"
```

Use `--overwrite` to replace existing output files.
