import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

print("Loading local parquet file...")
# Load local top 500 audios dataset directly
df_top = pd.read_parquet("hophacks_2026_10M_processed\\top_500_audios.parquet")

# Ensure required columns exist and drop missing values
df_top = df_top.dropna(subset=['music_id', 'create_time'])

# Ensure create_time is in datetime format
if not pd.api.types.is_datetime64_any_dtype(df_top['create_time']):
    df_top['create_date'] = pd.to_datetime(df_top['create_time'], unit='s')
else:
    df_top['create_date'] = df_top['create_time']

# Compute top 500 audios ordered by overall frequency (Rank #1 to Rank #500)
top_500_audios = df_top['music_id'].value_counts().nlargest(4).index

# ==============================================================================
# PLOT 1: Absolute Calendar Timeline (Rank-Based Gradient)
# Pure Green = Rank #1 Audio | Pure Blue = Rank #500 Audio
# ==============================================================================
print("\n--- Processing Plot 1: Calendar Timeline ---")

# Round down creation date to start of the week for smoother line aggregation
df_top['week'] = df_top['create_date'].dt.to_period('W').dt.start_time

# Pivot so Rows = Weeks, Columns = music_id, Values = Video count
pivot_df_1 = df_top.groupby(['week', 'music_id']).size().unstack(fill_value=0)

# Reorder pivot columns explicitly from Rank #1 to Rank #500
pivot_df_1 = pivot_df_1.reindex(columns=top_500_audios)

# Define Green (#1) to Blue (#500) Colormap
green_to_blue = LinearSegmentedColormap.from_list("GreenToBlue", ["green", "blue"])

plt.figure(figsize=(14, 7))
ax1 = plt.gca()

num_tracks_1 = len(pivot_df_1.columns)
for i, col in enumerate(pivot_df_1.columns):
    # Normalized rank: 0.0 for rank #1 (Green), 1.0 for rank #500 (Blue)
    color = green_to_blue(i / max(1, num_tracks_1 - 1))
    ax1.plot(pivot_df_1.index, pivot_df_1[col], alpha=0.3, color=color, linewidth=1.2)

plt.title('Trending Lifespans: Top 4 TikTok Audios Over Time (Green = #1, Blue = #4)', fontsize=14, pad=15)
plt.xlabel('Creation Date', fontsize=12)
plt.ylabel('Videos Posted per Week', fontsize=12)

# Clean up visual aesthetics
plt.grid(True, axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# ==============================================================================
# PLOT 2: Normalized Relative Trajectory (Recency-Based Gradient)
# Pure Green = Most Recent First-Seen Date | Pure Blue = Oldest First-Seen Date
# ==============================================================================
print("\n--- Processing Plot 2: Relative 3-Month Trajectory ---")

# 1. Determine birth date (earliest create_date) per music_id
birthdates = df_top.groupby('music_id')['create_date'].min().reset_index()
birthdates.rename(columns={'create_date': 'birth_date'}, inplace=True)

# 2. Merge birth dates back into the main DataFrame
if 'birth_date' in df_top.columns:
    df_top = df_top.drop(columns=['birth_date'])
df_top = df_top.merge(birthdates, on='music_id')

# 3. Calculate relative age metrics (days and weeks since birth)
df_top['days_since_birth'] = (df_top['create_date'] - df_top['birth_date']).dt.days
df_top['week_index'] = df_top['days_since_birth'] // 7

# 4. Filter for the first 3 months (~12 weeks) post-launch
df_first_3_months = df_top[df_top['week_index'] <= 12]

# 5. Pivot so Rows = week_index (0–12), Columns = music_id, Values = Video count
pivot_df_2 = df_first_3_months.groupby(['week_index', 'music_id']).size().unstack(fill_value=0)

# 6. Map recency ranks: Oldest birth date = Blue (0.0), Most recent birth date = Green (1.0)
birthdates_sorted = birthdates.sort_values('birth_date')
birth_rank_map = {
    row['music_id']: i / max(1, len(birthdates_sorted) - 1) 
    for i, (_, row) in enumerate(birthdates_sorted.iterrows())
}

# Define Blue (Oldest) to Green (Most Recent) Colormap
blue_to_green = LinearSegmentedColormap.from_list("BlueToGreen", ["blue", "green"])

plt.figure(figsize=(12, 6))
ax2 = plt.gca()

for col in pivot_df_2.columns:
    recency_norm = birth_rank_map.get(col, 0.5)  # 0.0 = Oldest (Blue), 1.0 = Newest (Green)
    color = blue_to_green(recency_norm)
    ax2.plot(pivot_df_2.index, pivot_df_2[col], alpha=0.3, color=color, linewidth=1.2)

plt.title('Viral Trajectory: First 3 Months (Green = Most Recent, Blue = Oldest)', fontsize=14, pad=15)
plt.xlabel('Weeks Since Audio First Appeared', fontsize=12)
plt.ylabel('New Videos Posted per Week', fontsize=12)

# Lock x-axis to exactly 0–12 weeks
plt.xlim(0, 12)
plt.xticks(range(0, 13))

# Clean up visual aesthetics
plt.grid(True, axis='both', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

print(" Done processing and plotting!")