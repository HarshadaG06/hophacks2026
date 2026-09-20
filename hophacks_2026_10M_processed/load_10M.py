from datasets import load_dataset

# 1. Define your exact list of columns
columns_to_keep = [
    "id", "create_time", "desc", "stats_time", "play_count", 
    "digg_count", "share_count", "collect_count", "music_id", 
    "music_title", "music_author_name", "music_duration", 
    "music_original", "user_id", "user_verified", "country_code", "challenges"
]

print("Loading dataset...")
ds_dict = load_dataset("The-data-company/TikTok-10M")
ds = ds_dict["train"]

print("Dropping unused columns to save memory...")
# 2. Slice the dataset before doing any heavy processing
ds_slim = ds.select_columns(columns_to_keep)

print("Converting to Pandas DataFrame...")
# 3. Convert only the slimmed-down data into Pandas
df = ds_slim.to_pandas()

print(f"Success! DataFrame loaded with shape: {df.shape}")
print(df.head())

# 1. Print overall size and column names
print("=== DATASET OVERVIEW ===")
print(ds)

# 2. Print exact data types for every column
print("\n=== COLUMN DATA TYPES ===")
for col_name, col_type in ds.features.items():
    # .dtype shows the underlying type (int64, string, float32, etc.)
    print(f"{col_name}: {getattr(col_type, 'dtype', type(col_type))}")

# 3. Table of the first 5 rows
print("\n=== QUICK PREVIEW (First 5 Rows) ===")
# Taking a tiny slice and converting just that slice to Pandas is 100% RAM safe
sample_df = ds.select(range(5)).to_pandas()

# Use print() if in a terminal, or display() if you are using a Jupyter Notebook
print(sample_df)