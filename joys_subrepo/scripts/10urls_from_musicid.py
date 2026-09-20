import polars as pl

def get_sample_urls(music_id_target, input_file="greater_1000_output_videos.parquet", n_samples=10):
    """
    Scans the parquet dataset, filters by music_id, and prints/returns 
    up to `n_samples` URLs.
    """
    # Ensure numeric comparison matching the Parquet schema
    target_val = float(music_id_target)

    # Fetch top N matching records for 'url'
    urls_df = (
        pl.scan_parquet(input_file)
        .filter(pl.col("music_id") == target_val)
        .select("url")
        .limit(n_samples)
        .collect()
    )

    urls_list = urls_df["url"].to_list()
    
    print(f"Found {len(urls_list)} URL(s) for music_id {music_id_target}:\n")
    for i, url in enumerate(urls_list, 1):
        print(f"{i}. {url}")
        
    return urls_list

if __name__ == "__main__":
    TARGET_MUSIC_ID = 7142169219956738048 
    
    urls = get_sample_urls(
        music_id_target=TARGET_MUSIC_ID,
        input_file="greater_1000_videos.parquet",
        n_samples=10
    )