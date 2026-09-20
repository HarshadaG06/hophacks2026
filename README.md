# Audio lifespan explorer



For each shortlisted audio: **when within its lifespan are the most-viewed videos concentrated**, and how does posting burstiness relate to that concentration?



Uses real data only: `music_id`, `post_date`, `total_views` (final count at snapshot) via `pipeline/loader.py`. No decay assumption, no estimated watch series.



## Setup



```powershell

cd D:\coding\hophacks2026

uv venv

.\.venv\Scripts\Activate.ps1

uv pip install -r requirements.txt

```



Set `SNAPSHOT_DATE` in `loader.py` if you know the collection date (defaults to `max(post_date)`).



## Run order



```powershell

python pipeline/s3_timeseries.py

python pipeline/s4_lifespan.py

python pipeline/s5_burstiness.py

python pipeline/s6_explore.py

python pipeline/export_json.py

cd viz

python -m http.server

```



Open http://localhost:8000

See **[viz/GRAPH_GUIDE.md](viz/GRAPH_GUIDE.md)** for an explanation of every chart.



Deprecated assumed-decay scripts live in `pipeline/deprecated/` (not part of the main pipeline).



## Burstiness (core)



### Kleinberg (2002) on posting events



Per audio, video post timestamps (day offsets from `audio_start`) are fed to Kleinberg's burst detection via vendored `pipeline/kleinberg.py` (MIT port of [pybursts](https://github.com/romain-fontugne/pybursts)).



| Parameter | Value | Rationale |

|-----------|-------|-----------|

| `KLEINBERG_S` | 2 | Standard default; controls cost of adding burst levels |

| `KLEINBERG_GAMMA` | 1.0 | Standard default; state-transition penalty |

| `MIN_EVENTS` | 20 | Skip Kleinberg features below this (NaN) |

| `BURST_LEVEL_MIN` | 1 | Count bursts at level ≥ 1 |

| `JITTER_SEED` | 0 | Same-day posts get small uniform jitter so gaps ≠ 0 |



pybursts 0.1.1 computes `n = len(gaps)` and `T = sum(gaps)` internally per audio — there are no optional `n`/`T` args. We use **identical s and gamma across all audios** so the fixed cost function is comparable (Kleinberg Eq. 3).



### Burst features (in `lifespan_per_audio.parquet`)



| Feature | Meaning |

|---------|---------|

| `kleinberg_max_level` | Deepest nested burst level |

| `kleinberg_n_bursts` | Intervals at level ≥ 1 |

| `top_burst_start_u` / `top_burst_end_u` | Highest-level burst (longest if tied) |

| `top_burst_length_share` | Top burst duration / lifespan |

| `share_of_videos_in_bursts` | Fraction of videos posted inside bursts |

| `share_of_views_in_bursts` | Fraction of total views from burst-period posts |

| `share_of_views_in_top_burst` | Views in the top burst only |

| `top_video_in_burst_share` | Fraction of top-decile-by-views videos posted in bursts |

| `burst_views_centroid_u` | Views-weighted mean u of burst-period posts |

| `views_burst_lift` | `share_of_views_in_bursts / share_of_videos_in_bursts` (>1 = bursts concentrate views) |



### Sensitivity



`s6_explore.py` reruns Kleinberg on 100 random audios for `gamma ∈ {0.5, 1.0, 2.0}` and `s ∈ {2, 3}` (parallelized, events capped at 3000 for speed), then prints median Spearman correlation of `kleinberg_max_level` and `share_of_views_in_bursts` across settings. High correlation → conclusions are robust to parameter choice.



## Lifespan metrics



| Metric | Meaning |

|--------|---------|

| `views_centroid_u` | Weighted mean lifespan position of views (0 = start, 1 = end) |

| `count_centroid_u` | Mean lifespan position of video posts |

| `centroid_gap` | `views_centroid_u − count_centroid_u`; positive = big videos land later |

| `top_video_centroid_u` | Mean position of top-decile videos by views |

| `share_views_first25` / `last25` | Share of total views in first/last 25% of lifespan |

| `age_views_spearman` | Within-audio correlation of video age vs views (age bias check) |



## Age-bias caveat



`total_views` is a **snapshot** count. Older videos have had more time to accumulate views. Check `age_views_spearman` and the `_old` / `_log` robustness columns.



## Outputs



| Step | Writes |

|------|--------|

| `s4_lifespan.py` | `lifespan_daily.parquet`, `lifespan_bins.parquet`, `lifespan_per_audio.parquet` |

| `s5_burstiness.py` | `kleinberg_bursts.parquet`, updates `lifespan_per_audio.parquet` with burst columns |

| `s6_explore.py` | `audio_features.parquet`, `lifespan_explore_meta.json` |

| `export_json.py` | `viz/data/*.json` including `kleinberg_bursts.json` |



## Viz



- Daily bar charts with Kleinberg burst bands (level-colored) and views centroid line

- Normalized lifespan curves, centroid histogram/scatter, mean curves by lifespan tercile

- Kleinberg max-level histogram

- `burst_views_centroid_u` vs `views_centroid_u` scatter, linked brushing with other panels

