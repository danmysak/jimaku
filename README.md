# jimaku 字幕

Generate Japanese learning subtitles from video files. Each subtitle entry contains three lines:

1. **Japanese** (kanji) — corrected transcription
2. **Hiragana** — full kana reading
3. **English** — natural translation

Designed for Japanese learners watching movies, shows, and other video content.

## Example Output

```srt
1
00:13:00,980 --> 00:13:12,939
どうした？ん？おい。よし。どこまで来たの？
どうした？ん？おい。よし。どこまできたの？
What's wrong? Hm? Hey. Okay. How far did you get?
```

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) and ffprobe on PATH
- API keys (set in `.env` or environment):
  - `OPENAI_API_KEY` — for GPT translation
  - `ELEVENLABS_API_KEY` — for Scribe ASR (default engine)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Basic — uses ElevenLabs Scribe + GPT-4.1
python jimaku.py movie.mkv

# Specify output path
python jimaku.py movie.mkv -o movie.jimaku.srt

# Provide English subtitles for better translation
python jimaku.py movie.mkv --english-srt movie.en.srt

# Use OpenAI Whisper instead of Scribe
python jimaku.py movie.mkv --asr whisper

# Use a different translation model
python jimaku.py movie.mkv --translation-model gpt-4o
```

If no English subtitles are provided, the tool automatically extracts embedded English subtitle tracks from the video file (if available).

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `<input>.srt` | Output SRT path |
| `--asr` | `scribe` | ASR engine: `scribe` (ElevenLabs) or `whisper` (OpenAI) |
| `--translation-model` | `gpt-4.1` | GPT model for kana + translation |
| `--english-srt` | — | Path to English reference subtitles |
| `--chunk-duration` | `600` | Audio chunk length in seconds (whisper only) |
| `--batch-size` | `30` | Segments per translation API call |

## Cost

Processing a 2-hour Japanese movie (Perfect Days, 2023) cost **~$1.03**:

| Component | Cost |
|-----------|------|
| ElevenLabs Scribe (124.5 min) | $0.83 |
| GPT-4.1 translation (13 batches) | $0.20 |
| **Total** | **$1.03** |

The tool reports progress and costs as it runs.

## How It Works

1. **Extract audio** — ffmpeg extracts mono 16kHz MP3 from the video
2. **Transcribe** — ElevenLabs Scribe (or OpenAI Whisper) produces word-level timestamps, grouped into subtitle segments
3. **Translate** — GPT corrects transcription errors, generates hiragana readings, and translates to English
4. **Output** — writes a standard SRT file with three lines per entry

## Note

This project was vibe-coded with [GitHub Copilot](https://github.com/features/copilot).

## License

MIT
