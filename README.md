# jimaku 字幕

Generate Japanese learning subtitles from video files. Each subtitle entry contains three lines:

1. **Japanese** — transcription with kanji
2. **Hiragana** — full kana reading
3. **English** — translation (from existing English subtitles)

Designed for Japanese learners watching movies, shows, and other video content.

## Example Output

```srt
19
00:13:12,940 --> 00:13:19,500
大丈夫。よし。
だいじょうぶ。よし。
It's gonna be fine. Alright.

24
00:14:48,080 --> 00:15:00,000
ドアが閉まります。ご注意ください。
どあがしまります。ごちゅういください。
The doors are closing. Please be careful.
```

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) and ffprobe on PATH
- English subtitles (embedded in the video or supplied separately)
- API keys (set in `.env` or environment):
  - `OPENAI_API_KEY` — for LLM matching (GPT-5.2)
  - `ELEVENLABS_API_KEY` — for Scribe ASR

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Basic — uses embedded English subs as guide
python jimaku.py movie.mkv

# Specify output path
python jimaku.py movie.mkv -o movie.jimaku.srt

# Provide external English subtitles
python jimaku.py movie.mkv --english-srt movie.en.srt

# Use a different LLM model
python jimaku.py movie.mkv --model gpt-4.1

# Disable reasoning (faster, cheaper, less accurate)
python jimaku.py movie.mkv --thinking low

# Save debug info (API inputs/outputs, segments)
python jimaku.py movie.mkv --debug-dir ./debug
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `<input>.srt` | Output SRT path |
| `--model` | `gpt-5.2` | LLM for matching and kana generation |
| `--thinking` | `medium` | Reasoning effort: `low`, `medium`, `high` |
| `--english-srt` | — | Path to English SRT (default: auto-extract from video) |
| `--batch-size` | `30` | Max English subs per LLM call |
| `--debug-dir` | — | Directory to save debug info |

## Cost

Processing a 2-hour Japanese movie (Perfect Days, 2023) costs **~$0.96**:

| Component | Cost |
|-----------|------|
| ElevenLabs Scribe (31.9 min of fragments) | $0.21 |
| GPT-5.2 matching w/ medium reasoning (78 batches) | $0.75 |
| **Total** | **$0.96** |

The tool reports progress and costs as it runs.

## How It Works

1. **Extract English subtitles** — from the video file or a supplied SRT
2. **Extract audio** — ffmpeg extracts mono 16 kHz WAV
3. **Subtitle-guided transcription** — English subs are clustered by timing; short audio fragments are extracted around each cluster and transcribed with ElevenLabs Scribe
4. **LLM matching** — Japanese ASR text (with word-level timestamps) is sent to an LLM alongside English subtitles; the model aligns them by timestamp, adds hiragana readings, and fixes minor ASR errors
5. **Output** — writes a standard SRT file with three lines per entry, using English subtitle timestamps for perfect timing

## Note

This project was vibe-coded with [GitHub Copilot](https://github.com/features/copilot).

## License

MIT
