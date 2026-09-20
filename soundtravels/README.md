# SoundTravels

An interactive story about how TikTok audios move across hashtag and broader topic-family landscapes over time. The current data spans **2025-01-01 through 2025-06-17** inside the fixed 2025 observation frame.

The visualization contains no simulated or placeholder points. It loads the checked-in browser export derived from `final.parquet`, defaults to the most-used plotted audio, and synchronizes daily movement and comparison-window controls across two fixed semantic maps and a usage chart.

## Run locally

From this folder:

```powershell
node serve.mjs
```

Then open <http://127.0.0.1:4173>.

There is no install or build step. The site uses plain HTML, CSS, JavaScript, and SVG.

## Project structure

```text
dist/
  index.html                  Page structure and narrative
  styles.css                  Visual design and responsive layout
  app.mjs                     Controls and observed-data loading state
  assets/
    audio/                    Optional playable audio files
  data/
    umap-data.json            Observed browser payload derived from the parquet
.openai/hosting.json          Existing private Sites deployment binding
DATA_CONTRACT.md              Required real-data export schema
serve.mjs                     Small local static server
```

## Refresh the real data

Read `DATA_CONTRACT.md`. Place `final.parquet` in this folder, then run:

```powershell
python prepare_data.py
```

The preparation step uses only unique rows with observed posting times and valid coordinates in both UMAPs. When playable files are present in `dist/assets/audio/`, it populates the selector with those audio IDs, ordered by observed plotted-use count; otherwise it falls back to the five highest-use audios.

See `AUDIO_FILES.md` for the available TikTok identifiers and representative post URLs. The parquet does not include playable media; adding a supported file under `dist/assets/audio/<music_id>.<extension>` and rerunning the preparation script enables the corresponding Play Audio button.

## Version control and deployment

This directory is already a Git repository. To move it to a new remote:

```powershell
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

If `origin` already exists, update it with `git remote set-url origin <YOUR_REPOSITORY_URL>`.

For another Sites deployment, replace or remove `.openai/hosting.json` so the new owner does not reuse the existing private project binding. Other static hosts can publish `dist/` directly.

No dataset, API key, credential, dependency folder, or generated synthetic data is included.
