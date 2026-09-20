"""Build the browser payload from final.parquet without inventing records."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "final.parquet"
OUTPUT = ROOT / "dist" / "data" / "umap-data.json"
AUDIO_DIRECTORY = ROOT / "dist" / "assets" / "audio"
WINDOW_START = pd.Timestamp("2025-01-01", tz="UTC")
WINDOW_END = pd.Timestamp("2025-07-01", tz="UTC")
TOP_AUDIO_COUNT = 5

COLUMNS = [
    "id", "music_id", "create_time", "view_count", "url",
    "hash_umap_x", "hash_umap_y", "caption_umap_x", "caption_umap_y",
    "hash_topic_id", "hash_topic_keywords",
    "caption_topic_id", "caption_topic_keywords",
]


def stable_id(value: float | int) -> str:
    return format(float(value), ".0f")


def family_label(topic_id: int, keywords: object) -> str:
    if int(topic_id) < 0 or pd.isna(keywords):
        return "Unassigned / outlier"
    words = [word.strip() for word in str(keywords).split(",") if word.strip()]
    return ", ".join(words[:4]) or f"Family {int(topic_id)}"


def local_audio_file(music_id: str) -> str | None:
    for extension in ("mp3", "m4a", "ogg", "wav", "mp4"):
        candidate = AUDIO_DIRECTORY / f"{music_id}.{extension}"
        if candidate.is_file():
            return f"./assets/audio/{candidate.name}"
    return None


def main() -> None:
    frame = pq.read_table(SOURCE, columns=COLUMNS).to_pandas()
    dates = pd.to_datetime(frame["create_time"], unit="s", utc=True)
    in_window = (dates >= WINDOW_START) & (dates < WINDOW_END)
    coordinates = ["hash_umap_x", "hash_umap_y", "caption_umap_x", "caption_umap_y"]
    frame = frame.loc[in_window].dropna(subset=coordinates).drop_duplicates(subset="id").copy()
    frame["observed_at"] = pd.to_datetime(frame["create_time"], unit="s", utc=True)
    frame["music_key"] = frame["music_id"].map(stable_id)

    ranked = frame.groupby("music_key").size().sort_values(ascending=False)
    playable_ids = [music_id for music_id in ranked.index if local_audio_file(music_id)]
    ranked = ranked.loc[playable_ids].head(TOP_AUDIO_COUNT) if playable_ids else ranked.head(TOP_AUDIO_COUNT)
    audios = []
    for music_id, observed_uses in ranked.items():
        rows = frame[frame["music_key"] == music_id]
        source_rows = rows.dropna(subset=["url"]).sort_values(["view_count", "create_time"], ascending=[False, True])
        source_url = None if source_rows.empty else str(source_rows.iloc[0]["url"])
        audios.append({
            "music_id": music_id,
            "label": f"Audio {music_id} · {int(observed_uses):,} uses",
            "observed_uses": int(observed_uses),
            "source_video_id": None if source_rows.empty else str(int(source_rows.iloc[0]["id"])),
            "source_video_url": source_url,
            "audio_file": local_audio_file(music_id),
            "expected_audio_file": f"assets/audio/{music_id}.mp3",
        })

    hashtag_families, topic_families, videos = {}, {}, []
    for row in frame.itertuples(index=False):
        hash_id, topic_id = str(int(row.hash_topic_id)), str(int(row.caption_topic_id))
        hashtag_families.setdefault(hash_id, family_label(row.hash_topic_id, row.hash_topic_keywords))
        topic_families.setdefault(topic_id, family_label(row.caption_topic_id, row.caption_topic_keywords))
        videos.append({
            "video_id": str(int(row.id)), "music_id": row.music_key, "ts": int(row.create_time),
            "hx": round(float(row.hash_umap_x), 4), "hy": round(float(row.hash_umap_y), 4),
            "tx": round(float(row.caption_umap_x), 4), "ty": round(float(row.caption_umap_y), 4),
            "hf": hash_id, "tf": topic_id,
        })

    payload = {
        "schemaVersion": 5,
        "window": {
            "start": frame["observed_at"].min().floor("D").isoformat(),
            "end": frame["observed_at"].max().floor("D").isoformat(),
            "bin": "day",
        },
        "defaultAudioId": audios[0]["music_id"],
        "audios": audios,
        "families": {"hashtag": hashtag_families, "topic": topic_families},
        "videos": videos,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(videos):,} observed videos to {OUTPUT}")
    for audio in audios:
        print(f"{audio['music_id']}: {audio['observed_uses']:,} uses; source={audio['source_video_url']}; file={audio['audio_file']}")


if __name__ == "__main__":
    main()
