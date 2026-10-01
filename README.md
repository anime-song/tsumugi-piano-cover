# tsumugi-piano-cover

**Generate piano covers of pop songs as MIDI.** The original song is first transcribed into a multi-instrument MIDI with [tsumugi](https://github.com/anime-song/tsumugi), and a piano model pretrained on about 3,000 hours of piano performances generates the cover from it.

[日本語 README](README_ja.md) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) | [![Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/anime-song/tsumugi-piano-cover) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anime-song/tsumugi-piano-cover/blob/main/notebooks/cover_studio_colab_en.ipynb)

- 🎹 **Covers on the original's time axis**: play the cover together with the original recording
- 🎚️ **Controllable arrangement**: faithfulness to the original, dynamics, fills, range, and performer style
- 🔁 **Cover Studio**: a web UI for regenerating with different settings, comparing takes, and regenerating from any point

---

## Quick start

### Google Colab

The easiest way is the Colab notebook ([English](https://colab.research.google.com/github/anime-song/tsumugi-piano-cover/blob/main/notebooks/cover_studio_colab_en.ipynb) / [日本語](https://colab.research.google.com/github/anime-song/tsumugi-piano-cover/blob/main/notebooks/cover_studio_colab_ja.ipynb)). Select a GPU runtime, run the first cell, and click **"▶ Open Cover Studio"**. Upload a song, then transcribe, generate, compare, and download everything in the browser.

### Local

Requires Python 3.12 and an NVIDIA GPU (a CPU works, but slowly). The requirements install PyTorch 2.13 with CUDA 13.0.

```bash
git clone https://github.com/anime-song/tsumugi-piano-cover.git
cd tsumugi-piano-cover
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
.venv/bin/python -m cover_studio              # opens http://127.0.0.1:8100
```

The model weights are downloaded from Hugging Face the first time you generate a cover. The built UI is downloaded from GitHub Releases, so Node.js is not required.

To transcribe audio inside Cover Studio, set up tsumugi in its own environment at `.tsumugi/tsumugi`:

```bash
git clone https://github.com/anime-song/tsumugi.git .tsumugi/tsumugi
cd .tsumugi/tsumugi
uv sync --locked --extra stem
```

Cover Studio finds it automatically and runs it with tsumugi's own Python (`--tsumugi-dir` sets another location). On Windows, torchcodec also needs the shared FFmpeg libraries; put them in `.tsumugi/ffmpeg-shared/bin` or pass `--ffmpeg-dir`. Without tsumugi, you can still upload a MIDI file you have already transcribed, together with the audio.

---

## Cover Studio

![Cover Studio](docs/images/cover_studio.png)

A song is transcribed only once. Generation reuses the loaded model, so you can change settings and generate again as often as you like.

- **Takes**: every take keeps its settings, seed, and model. "Reuse settings" loads them back into the Create panel. Chips show only the settings that differ from the defaults
- **Regenerate from here**: keep the take up to the playhead and regenerate only the rest, with new settings
- **Compare**: the original audio and the cover play on the same clock. Switching takes (↑ / ↓) keeps the playback position. Use the piano roll with the source MIDI overlaid, loop regions, and the original / cover balance slider
- **Live transcription**: notes appear in the piano roll as tsumugi transcribes them, stem by stem
- **Export**: MIDI, or WAV rendered with piano samples in the browser
- Favorites, notes, performer presets, and Japanese / English UI

```bash
python -m cover_studio --root D:/covers --port 8080 --no-browser
python -m cover_studio --device cpu
```

---

## Python API and CLI

```python
from piano_cover.generate import CoverParams, generate_covers, prepare_source
from piano_cover.model import CoverModel

model = CoverModel.from_pretrained(device="cuda")  # anime-song/tsumugi-piano-cover
source = prepare_source("song.mid", model.tokenizer, model.config)  # tsumugi MIDI of the original
cover = generate_covers(model, source, CoverParams(seconds=60, fill=2.5))[0]
model.tokenizer.events_to_midi(cover.events, "cover.mid")
```

```bash
python -m piano_cover.generate --checkpoint anime-song/tsumugi-piano-cover --source song.mid --seconds 60
python -m piano_ar.generate --checkpoint anime-song/tsumugi-piano-cover --seconds 60   # solo piano, no source
```

The source MIDI must come from tsumugi with stem separation, instrument refinement, velocity, and beat/chord/key estimation enabled, the same settings used for training. [`cover_studio/tsumugi_worker.py`](cover_studio/tsumugi_worker.py) runs exactly this pipeline.

### Generation settings

| Setting | Default | Effect |
| --- | --- | --- |
| `source_cfg` | 2.0 | Guidance on the original. Higher follows the melody and chords more closely |
| `onset_bias` | 0.0 | Pulls note onsets toward the original's onsets (around 4 works well) |
| `dynamics` / `density` | 2.0 / 1.0 | Scales the loudness and note-density curves predicted from the original |
| `fill` / `above` / `span` | 2.0 / 0.0 / 1.5 | Shifts fills, notes above the melody, and range (in standard deviations) |
| `channel` / `channel_cfg` | 0 / 3.0 | Performer index (0 = unspecified) and how strongly to follow it |
| `temperature` / `top_p` | 1.0 / 0.90 | Sampling |
| `seconds` | full song | Generate only the beginning (for quick previews) |

---

## How it works

![Architecture](docs/images/piano_cover_architecture_en.svg)

- The song is generated in **2-second patches**. A Global Transformer runs across patches and a Local Transformer generates the notes inside each patch (time, pitch, duration, velocity, pedal), at 10 ms resolution.
- The original is encoded as **tsumugi's multi-instrument MIDI** (notes of all instruments including drums, plus beats, chords, and keys), summarized per patch and encoded over the whole song.
- The decoder looks at the original through **gated cross-attention** (gates start at 0): Global sees the song encoding around the current time, and Local sees the individual source notes.
- A **Planner** predicts the loudness, density, and arrangement curves from the original. These curves condition generation and are what the dynamics and arrangement settings scale.
- Covers are **not time-stretched** to the original during training. The alignment between cover and original only decides where cross-attention looks, so alignment errors do not distort the rhythm.

![Training pipeline](docs/images/training_pipeline_v2_en.svg)

1. **Pretraining (`piano_ar`)**: about 3,000 hours of solo piano MIDI, including transcribed piano performances, PiJAMA, the piano side of Pop2Piano, PIAST, and MAESTRO.
2. **Cover fine-tuning (`piano_cover`)**: about 3,000 piano covers (about 190 hours) of about 780 songs, paired with tsumugi transcriptions of the originals.

The training data is not distributed.

---

## Repository layout

| Path | Contents |
| --- | --- |
| [`piano_ar/`](piano_ar) | Piano performance model (tokenizer, model, pretraining) |
| [`piano_cover/`](piano_cover) | Cover model (source encoder, Planner, data preparation, training, generation) |
| [`cover_studio/`](cover_studio) | Cover Studio server (FastAPI) and the tsumugi worker |
| [`web/`](web) | Cover Studio UI (React + Vite). `npm run dev` / `npm run build` |
| [`notebooks/`](notebooks) | Colab notebooks |
| [`piano_score/`](piano_score) | Performance-to-score model (work in progress) |

To export your own training checkpoint in the published format, run `python -m piano_cover.export --checkpoint <best.pt> --out <dir>` (and `piano_ar.export` for the piano model).

---

## Limitations

- Errors in the tsumugi transcription (melody, beats, chords) carry over into the cover. The rhythm depends strongly on the beats in the source MIDI
- Swing is often played straight
- The sustain pedal tends to be held longer than in human performances
- The output is a MIDI performance, not a score

## License

- Code: [MIT License](LICENSE)
- Model weights ([Hugging Face](https://huggingface.co/anime-song/tsumugi-piano-cover)): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use only)

## Acknowledgements

- [tsumugi](https://github.com/anime-song/tsumugi) — multi-instrument transcription of the original songs
- [smplr](https://github.com/danigb/smplr) — piano samples for playback in Cover Studio
- Cover Studio follows the structure of [audio2chordpro](https://github.com/anime-song/audio2chordpro)
