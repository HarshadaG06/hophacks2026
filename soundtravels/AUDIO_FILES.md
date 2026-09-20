# SoundTravels audio files

The parquet contains `music_id` values and TikTok post-page URLs. It does **not** contain direct media URLs or playable audio/video bytes. The stored URLs use the form `https://tiktok.com/@/video/<video_id>` and could not be reliably retrieved as media, so no audio was downloaded automatically.

| Rank | Music ID | Observed plotted uses | Representative dataset video | File that enables Play Audio |
|---:|---|---:|---|---|
| 1 | `7437803750033868800` | 413 | [video 7458783812332424478](https://tiktok.com/@/video/7458783812332424478) | `dist/assets/audio/7437803750033868800.mp3` |
| 2 | `7474209817062575104` | 380 | [video 7483290131089198379](https://tiktok.com/@/video/7483290131089198379) | `dist/assets/audio/7474209817062575104.mp3` |
| 3 | `7072513628145977344` | 318 | [video 7503011072366955798](https://tiktok.com/@/video/7503011072366955798) | `dist/assets/audio/7072513628145977344.mp3` |
| 4 | `7289503994026805248` | 317 | [video 7507406755958279470](https://tiktok.com/@/video/7507406755958279470) | `dist/assets/audio/7289503994026805248.mp3` |
| 5 | `7488094138099436544` | 305 | [video 7495140651999300894](https://tiktok.com/@/video/7495140651999300894) | `dist/assets/audio/7488094138099436544.mp3` |

Manually obtain each sound from an authorized source and save it under the matching music ID in `dist/assets/audio/`. `prepare_data.py` also recognizes `.m4a`, `.ogg`, `.wav`, and `.mp4`. Run the script again after adding files; the corresponding Play Audio buttons will become active.
