# Final data handoff

The website is built from `final.parquet` with `prepare_data.py`. The source parquet remains local and is ignored by Git; the generated browser payload is `dist/data/umap-data.json`.

## Source columns used

```text
id
music_id
create_time
hash_umap_x
hash_umap_y
caption_umap_x
caption_umap_y
hash_topic_id
hash_topic_keywords
caption_topic_id
caption_topic_keywords
```

The export keeps only videos posted in the fixed 2025 observation frame that have valid coordinates in both maps. This guarantees that both UMAPs show the same video records and time window.

- `hash_umap_x/y` drive the Hashtag UMAP.
- `caption_umap_x/y` drive the Topic-family UMAP.
- `hash_topic_*` and `caption_topic_*` provide stable family IDs and visible labels.
- `create_time` supplies the observed daily bins and usage counts.

When matching playable files are present in `dist/assets/audio/`, the preparation script selects those `music_id` values, ordered by observed plotted-video counts. Otherwise it selects the five highest-use audios. The most-used selected audio loads by default. The parquet's `url` column supplies representative TikTok post-page URLs, but no direct media files; see `AUDIO_FILES.md` for the manual playback-file mapping.
