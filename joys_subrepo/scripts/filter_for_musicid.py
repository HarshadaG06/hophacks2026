import polars as pl

def export_videos_by_music_id(music_id_to_filter, input_file="greater_1000_videos.parquet", output_csv="music_id_export.csv"):
    """
    Scans the parquet dataset lazily, filters for matching music_id,
    and streams results directly into a CSV file.
    """
    (
        pl.scan_parquet(input_file)
        .filter(pl.col("music_id") == music_id_to_filter)
        .sink_csv(output_csv)
    )
    print(f"Exported rows matching music_id '{music_id_to_filter}' to '{output_csv}'")

if __name__ == "__main__":
    # Replace with the specific music_id you want to filter by
    TARGET_MUSIC_ID = 7142169219956738048 
    
    export_videos_by_music_id(
        music_id_to_filter=TARGET_MUSIC_ID,
        input_file="greater_1000_videos.parquet",
        output_csv=f"videos_music_{TARGET_MUSIC_ID}.csv"
    )