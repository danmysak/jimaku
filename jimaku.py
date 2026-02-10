#!/usr/bin/env python3
"""
jimaku — Generate Japanese learning subtitles from video/audio files.

For each subtitle segment produces three lines:
  1. Original Japanese text (kanji)
  2. Hiragana reading
  3. English translation

Uses OpenAI whisper-1 for timestamped transcription and a GPT model
to refine the text, add kana readings, and translate to English.
(gpt-4o-transcribe lacks timestamp support, so whisper-1 is used for
segmentation; GPT then corrects any recognition errors.)

Requirements:
  - ffmpeg and ffprobe on PATH
  - OPENAI_API_KEY environment variable
  - pip install openai

Usage:
  python jimaku.py movie.mkv
  python jimaku.py movie.mkv -o subs.srt
  python jimaku.py movie.mkv --english-srt movie.en.srt
  python jimaku.py movie.mkv --translation-model gpt-4o-mini
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package required.  pip install openai", file=sys.stderr)
    sys.exit(1)

try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenLabs = None  # optional; only needed with --asr scribe

# ── Pricing (USD, Feb 2026) ───────────────────────────────────────────────
TRANSCRIBE_COST_PER_MIN = {
    "whisper": 0.006,   # whisper-1
    "scribe":  0.0067,  # ElevenLabs Scribe
}

CHAT_PRICING = {                                       # per 1 M tokens
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
    "gpt-4.1":     {"input": 2.00,  "output": 8.00},
    "gpt-5":       {"input": 1.25,  "output": 10.00},
    "gpt-5.1":     {"input": 1.25,  "output": 10.00},
    "gpt-5.2":     {"input": 1.75,  "output": 14.00},
}

# ── Defaults ──────────────────────────────────────────────────────────────
CHUNK_SECONDS = 600        # target chunk length; actual splits land on silences
TRANSLATE_BATCH = 30       # subtitle segments per translation call
MAX_RETRIES = 3
RETRY_DELAY = 5            # seconds


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def fatal(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def check_deps(asr: str) -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            fatal(f"{tool} not found on PATH. Install ffmpeg first.")
    if not os.environ.get("OPENAI_API_KEY"):
        fatal("OPENAI_API_KEY environment variable is not set.")
    if asr == "scribe" and not os.environ.get("ELEVENLABS_API_KEY"):
        fatal("ELEVENLABS_API_KEY environment variable is not set. "
              "Get one at https://elevenlabs.io")


def probe_duration(path: str) -> float:
    """Return duration of a media file in seconds."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        fatal(f"ffprobe failed on {path}")
    return float(json.loads(r.stdout)["format"]["duration"])


def extract_audio(video: str, out: str) -> None:
    """Extract mono 16 kHz / 64 kbps mp3 from a video file."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video,
         "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
         "-f", "mp3", out],
        capture_output=True,
    )
    if r.returncode != 0:
        fatal(f"ffmpeg audio extraction failed:\n{r.stderr.decode()[-500:]}")


def detect_silences(
    audio: str, noise_db: int = -30, min_dur: float = 0.5,
) -> list[tuple[float, float]]:
    """Return [(start, end), …] silence intervals detected by ffmpeg."""
    r = subprocess.run(
        ["ffmpeg", "-i", audio,
         "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", r.stderr)]
    ends   = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)",   r.stderr)]
    return list(zip(starts, ends[: len(starts)]))


def pick_split_points(
    silences: list[tuple[float, float]],
    total_dur: float,
    target_sec: int,
) -> list[float]:
    """Choose split points at silences closest to each target interval."""
    points: list[float] = []
    target = float(target_sec)
    window = target_sec * 0.3  # search ±30 % around target
    while target < total_dur - target_sec * 0.25:
        best, best_dist = target, float("inf")  # fallback: exact target
        for s_start, s_end in silences:
            mid = (s_start + s_end) / 2
            dist = abs(mid - target)
            if dist < best_dist and dist < window:
                best, best_dist = mid, dist
        points.append(best)
        target = best + target_sec
    return points


def split_audio(
    audio: str,
    chunk_dir: str,
    target_sec: int,
) -> list[tuple[str, float]]:
    """Split audio at detected silences.

    Returns [(chunk_path, chunk_start_seconds), …].
    """
    total_dur = probe_duration(audio)

    silences = detect_silences(audio)
    splits = pick_split_points(silences, total_dur, target_sec)

    boundaries = [0.0] + splits + [total_dur]
    chunks: list[tuple[str, float]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        out = os.path.join(chunk_dir, f"chunk_{i:04d}.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio,
             "-ss", str(start), "-to", str(end),
             "-c:a", "copy", out],
            capture_output=True, check=True,
        )
        chunks.append((out, start))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
#  SRT parsing (for English reference subtitles)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_srt_ts(ts: str) -> float:
    """Parse 'HH:MM:SS,mmm' → seconds."""
    m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", ts.strip())
    if not m:
        return 0.0
    h, mn, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600 + mn * 60 + s + ms / 1000


def parse_srt(path: str) -> list[dict]:
    """Parse an SRT file → list of {start, end, text}."""
    with open(path, encoding="utf-8-sig") as f:
        content = f.read()
    entries: list[dict] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # find the timestamp line (skip index line)
        ts_match = re.match(
            r"([\d:,.]+)\s*-->\s*([\d:,.]+)", lines[1] if len(lines) > 2 else lines[0],
        )
        if not ts_match:
            ts_match = re.match(r"([\d:,.]+)\s*-->\s*([\d:,.]+)", lines[0])
            text_lines = lines[1:]
        else:
            text_lines = lines[2:]
        if not ts_match:
            continue
        text = " ".join(ln.strip() for ln in text_lines if ln.strip())
        if text:
            entries.append({
                "start": _parse_srt_ts(ts_match[1]),
                "end":   _parse_srt_ts(ts_match[2]),
                "text":  text,
            })
    return entries


def find_overlapping_english(
    seg_start: float, seg_end: float, en_subs: list[dict],
) -> list[str]:
    """Return English subtitle texts that overlap the given time range."""
    result = []
    for e in en_subs:
        if e["start"] < seg_end and e["end"] > seg_start:
            result.append(e["text"])
    return result


def extract_embedded_subs(video: str, tmp_dir: str) -> str | None:
    """If *video* contains an English subtitle stream, extract it to a temp
    SRT file and return the path.  Returns None if no suitable stream found."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", video],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    streams = json.loads(r.stdout).get("streams", [])
    # prefer English text-based subtitle streams
    sub_streams = [
        s for s in streams
        if s.get("codec_type") == "subtitle"
        and s.get("codec_name") in ("subrip", "srt", "ass", "ssa", "mov_text")
    ]
    if not sub_streams:
        return None
    # pick the first English stream, or fall back to the first subtitle stream
    chosen = next(
        (s for s in sub_streams
         if s.get("tags", {}).get("language", "").startswith("en")),
        sub_streams[0],
    )
    out = os.path.join(tmp_dir, "ref_en.srt")
    idx = chosen["index"]
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-map", f"0:{idx}", out],
        capture_output=True,
    )
    if r.returncode != 0 or not os.path.isfile(out):
        return None
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  OpenAI calls (with retry)
# ═══════════════════════════════════════════════════════════════════════════

def _retry(fn, description: str):
    """Call *fn* up to MAX_RETRIES times with exponential back-off."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                fatal(f"{description} failed after {MAX_RETRIES} attempts: {exc}")
            wait = RETRY_DELAY * (2 ** (attempt - 1))
            print(f"       ⚠ {description} attempt {attempt} failed ({exc}), "
                  f"retrying in {wait}s …")
            time.sleep(wait)


def transcribe_chunk(client: OpenAI, path: str) -> list[dict]:
    """Transcribe one audio chunk → list of {start, end, text}.

    Uses whisper-1 with verbose_json for segment-level timestamps.
    (gpt-4o-transcribe does not support timestamps.)
    """
    def call():
        with open(path, "rb") as f:
            return client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language="ja",
            )

    resp = _retry(call, f"Transcribe {Path(path).name}")
    segments: list[dict] = []
    for seg in (resp.segments or []):
        text = seg.text.strip()
        # skip noise / empty segments
        if text and len(text) > 1:
            segments.append({"start": seg.start, "end": seg.end, "text": text})
    return segments


def transcribe_scribe(api_key: str, path: str) -> list[dict]:
    """Transcribe audio with ElevenLabs Scribe → list of {start, end, text}.

    Returns sentence-level segments built from word-level timestamps.
    """
    if ElevenLabs is None:
        fatal("elevenlabs package required for --asr scribe.  "
              "pip install elevenlabs")

    client = ElevenLabs(api_key=api_key)

    def call():
        with open(path, "rb") as f:
            return client.speech_to_text.convert(
                file=f,
                model_id="scribe_v2",
                language_code="jpn",
                tag_audio_events=False,
                timestamps_granularity="word",
            )

    resp = _retry(call, f"Scribe {Path(path).name}")

    # Group words into subtitle-sized segments by pauses and punctuation
    words = resp.words or []
    segments: list[dict] = []
    buf: list[dict] = []
    JP_SENT_END = set("。！？!?")

    for w in words:
        if getattr(w, "type", "word") != "word":
            continue
        text = w.text if isinstance(w, dict) else getattr(w, "text", "")
        start = w["start"] if isinstance(w, dict) else getattr(w, "start", 0)
        end = w["end"] if isinstance(w, dict) else getattr(w, "end", 0)
        if not text.strip():
            continue

        # decide whether to flush current buffer before adding this word
        if buf:
            gap = start - buf[-1]["end"]
            duration = end - buf[0]["start"]
            prev_text = buf[-1]["text"]
            sentence_break = prev_text and prev_text[-1] in JP_SENT_END
            if gap > 0.7 or duration > 12.0 or (sentence_break and gap > 0.15):
                seg_text = "".join(b["text"] for b in buf).strip()
                if seg_text:
                    segments.append({
                        "start": buf[0]["start"],
                        "end": buf[-1]["end"],
                        "text": seg_text,
                    })
                buf = []

        buf.append({"start": start, "end": end, "text": text})

    # flush remaining
    if buf:
        seg_text = "".join(b["text"] for b in buf).strip()
        if seg_text:
            segments.append({
                "start": buf[0]["start"],
                "end": buf[-1]["end"],
                "text": seg_text,
            })

    return segments


def translate_batch(
    client: OpenAI,
    texts: list[str],
    model: str,
    english_hints: list[str] | None = None,
) -> tuple[list[dict], int, int]:
    """
    Send a batch of Japanese lines → returns (results, prompt_tokens, completion_tokens).
    Each result: {"index": int, "japanese": str, "kana": str, "english": str}
    *english_hints* is an optional list (one per text) of reference English
    translations to guide correction and translation.
    """
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    hint_block = ""
    if english_hints:
        hint_lines = "\n".join(
            f"{i+1}. {h}" for i, h in enumerate(english_hints) if h
        )
        if hint_lines:
            hint_block = (
                "\n\n--- REFERENCE ENGLISH SUBTITLES (same numbering) ---\n"
                + hint_lines
            )

    en_rule = ""
    if english_hints:
        en_rule = (
            "• english: use the reference English subtitle as a starting point "
            "but adjust it so it accurately reflects the Japanese audio. "
            "If no reference is available for a line, translate from scratch.\n"
        )
    else:
        en_rule = "• english: a natural, concise translation.\n"

    system_prompt = (
        "You are a Japanese-language expert.\n"
        "The user provides numbered Japanese sentences from an "
        "automatic speech-recognition transcript (whisper)."
        + (
            " Reference English subtitles for the same scene are also "
            "provided — use them to better understand context, fix "
            "recognition errors, and improve translations."
            if english_hints else ""
        )
        + "\nFor each sentence, return a JSON object:\n"
        '{\"results\": [{\"index\": 1, '
        '\"japanese\": \"<corrected Japanese text>\", '
        '\"kana\": \"<full hiragana reading>\", '
        '\"english\": \"<natural English translation>\"}, …]}\n'
        "Rules:\n"
        "• japanese: fix any obvious recognition errors, wrong "
        "kanji, or garbled text while keeping the original "
        "meaning. If the text looks correct, keep it as-is.\n"
        "• kana: the complete hiragana reading of the corrected "
        "sentence (convert all kanji and katakana).\n"
        + en_rule
        + "• Preserve original ordering.\n"
        "• Return ONLY valid JSON, nothing else."
    )

    user_content = numbered + hint_block

    def call():
        return client.chat.completions.create(
            model=model,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

    resp = _retry(call, "Translation batch")
    data = json.loads(resp.choices[0].message.content)
    results = data.get("results", [])
    return results, resp.usage.prompt_tokens, resp.usage.completion_tokens


# ═══════════════════════════════════════════════════════════════════════════
#  SRT formatting
# ═══════════════════════════════════════════════════════════════════════════

def _ts(seconds: float) -> str:
    """Format seconds → SRT timestamp  HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{_ts(seg['start'])} --> {_ts(seg['end'])}\n")
            f.write(f"{seg['text']}\n")
            f.write(f"{seg.get('kana', '')}\n")
            f.write(f"{seg.get('english', '')}\n\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Cost helpers
# ═══════════════════════════════════════════════════════════════════════════

def translation_cost(model: str, in_tok: int, out_tok: int) -> float | None:
    """Return estimated cost in USD, or None if pricing is unknown."""
    p = CHAT_PRICING.get(model)
    if p is None:
        return None
    return (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate Japanese learning subtitles "
                    "(kanji + kana + English) from video/audio.",
    )
    ap.add_argument("video", help="Path to video or audio file")
    ap.add_argument("-o", "--output",
                    help="Output .srt path (default: <input>.srt)")
    ap.add_argument("--asr", default="scribe",
                    choices=["whisper", "scribe"],
                    help="ASR engine: whisper (OpenAI) or scribe "
                         "(ElevenLabs, better quality) (default: scribe)")
    ap.add_argument("--translation-model", default="gpt-4.1",
                    help="Model for kana/translation (default: gpt-4.1)")
    ap.add_argument("--chunk-duration", type=int, default=CHUNK_SECONDS,
                    help="Audio chunk length in seconds (default: 600)")
    ap.add_argument("--english-srt",
                    help="Path to English .srt subtitles (improves "
                         "transcription correction and translation)")
    ap.add_argument("--batch-size", type=int, default=TRANSLATE_BATCH,
                    help="Segments per translation API call (default: 30)")
    args = ap.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────
    video_path = args.video
    if not os.path.isfile(video_path):
        fatal(f"File not found: {video_path}")
    check_deps(args.asr)

    out_path = args.output or str(Path(video_path).with_suffix(".srt"))
    client = OpenAI()

    cost_transcribe = 0.0
    cost_translate = 0.0
    tok_in = 0
    tok_out = 0

    print(f"\n  jimaku — Japanese learning subtitle generator\n")

    with tempfile.TemporaryDirectory(prefix="jimaku_") as tmp:
        # ── Load English reference subs ───────────────────────────────────
        en_subs: list[dict] = []
        if args.english_srt:
            if not os.path.isfile(args.english_srt):
                fatal(f"English SRT not found: {args.english_srt}")
            en_subs = parse_srt(args.english_srt)
            print(f"  English reference: {len(en_subs)} subtitles "
                  f"loaded from {Path(args.english_srt).name}")
        else:
            # try to extract English subs embedded in the video
            embedded = extract_embedded_subs(video_path, tmp)
            if embedded:
                en_subs = parse_srt(embedded)
                print(f"  English reference: {len(en_subs)} subtitles "
                      f"extracted from video")
            else:
                print("  English reference: none (no embedded subs found)")

        # ── 1. Extract audio ──────────────────────────────────────────────
        audio = os.path.join(tmp, "audio.mp3")
        print(f"[1/3] Extracting audio from {Path(video_path).name} …")
        t0 = time.time()
        extract_audio(video_path, audio)
        dur_s = probe_duration(audio)
        dur_m = dur_s / 60
        print(f"       Duration: {dur_m:.1f} min  |  "
              f"Extracted in {time.time() - t0:.1f}s")

        # ── 2–3. Transcribe ───────────────────────────────────────────────
        asr = args.asr
        asr_cost_per_min = TRANSCRIBE_COST_PER_MIN[asr]
        all_segs: list[dict] = []

        if asr == "scribe":
            print(f"[2/3] Transcribing (ElevenLabs Scribe) …")
            t0 = time.time()
            all_segs = _retry(
                lambda: transcribe_scribe(
                    os.environ["ELEVENLABS_API_KEY"], audio,
                ),
                "Scribe transcription",
            )
            cost_transcribe = dur_m * asr_cost_per_min
            print(f"       {len(all_segs)} segments  ({time.time() - t0:.1f}s)  "
                  f"cost: ${cost_transcribe:.4f}")
        else:
            # whisper: split at silences, transcribe each chunk
            chunk_dir = os.path.join(tmp, "chunks")
            os.makedirs(chunk_dir)
            print(f"[2/3] Splitting & transcribing (whisper-1) …")
            chunks = split_audio(audio, chunk_dir, args.chunk_duration)
            print(f"       {len(chunks)} chunk(s)  "
                  f"(split at silence boundaries)")
            for ci, (cpath, chunk_start) in enumerate(chunks):
                t0 = time.time()
                segs = transcribe_chunk(client, cpath)
                for s in segs:
                    s["start"] += chunk_start
                    s["end"]   += chunk_start
                all_segs.extend(segs)
                chunk_min = probe_duration(cpath) / 60
                cost_transcribe += chunk_min * asr_cost_per_min
                print(f"       Chunk {ci + 1}/{len(chunks)}: "
                      f"{len(segs)} segments  ({time.time() - t0:.1f}s)  "
                      f"cost so far: ${cost_transcribe:.4f}")

        print(f"       ✓ {len(all_segs)} segments total  |  "
              f"Transcription cost: ${cost_transcribe:.4f}")

        if not all_segs:
            fatal("No speech segments detected.")

        # deduplicate whisper hallucinations (repeated identical text)
        before = len(all_segs)
        deduped: list[dict] = []
        for seg in all_segs:
            if deduped and seg["text"] == deduped[-1]["text"]:
                deduped[-1]["end"] = seg["end"]  # merge timespan
            else:
                deduped.append(seg)
        all_segs = deduped
        if len(all_segs) < before:
            print(f"       Merged {before - len(all_segs)} duplicate "
                  f"segment(s) → {len(all_segs)} remaining")

        # ── Translate + kana ───────────────────────────────────────────
        model = args.translation_model
        n_batches = math.ceil(len(all_segs) / args.batch_size)
        print(f"[3/3] Translating with {model}  "
              f"({n_batches} batch(es) of ≤{args.batch_size}) …")

        for bi in range(n_batches):
            t0 = time.time()
            lo = bi * args.batch_size
            hi = min(lo + args.batch_size, len(all_segs))
            batch_segs = all_segs[lo:hi]
            batch_texts = [s["text"] for s in batch_segs]

            # Build per-segment English hints from reference subs
            en_hints: list[str] | None = None
            if en_subs:
                en_hints = []
                for seg in batch_segs:
                    overlaps = find_overlapping_english(
                        seg["start"], seg["end"], en_subs,
                    )
                    en_hints.append(" | ".join(overlaps) if overlaps else "")

            results, p_tok, c_tok = translate_batch(
                client, batch_texts, model, english_hints=en_hints,
            )
            tok_in += p_tok
            tok_out += c_tok
            batch_cost = translation_cost(model, p_tok, c_tok)
            if batch_cost is not None:
                cost_translate += batch_cost

            # merge results back
            for r in results:
                idx = lo + r["index"] - 1
                if 0 <= idx < len(all_segs):
                    if r.get("japanese"):
                        all_segs[idx]["text"] = r["japanese"]
                    all_segs[idx]["kana"] = r.get("kana", "")
                    all_segs[idx]["english"] = r.get("english", "")

            cost_str = (f"${cost_translate:.4f}"
                        if cost_translate > 0 else "unknown (see tokens)")
            print(f"       Batch {bi + 1}/{n_batches}: "
                  f"{hi - lo} segments  ({time.time() - t0:.1f}s)  "
                  f"cost so far: {cost_str}")

        cost_str = (f"${cost_translate:.4f}"
                    if cost_translate > 0 else "unknown pricing")
        print(f"       ✓ Translation cost: {cost_str}")

    # ── Write SRT ─────────────────────────────────────────────────────────
    write_srt(all_segs, out_path)

    print()
    print("═" * 60)
    print(f"  Output file:       {out_path}")
    print(f"  Subtitle entries:  {len(all_segs)}")
    print(f"  Audio duration:    {dur_m:.1f} min")
    print(f"  ─────────────────────────────────────")
    print(f"  Transcription:     ${cost_transcribe:.4f}")
    if cost_translate > 0:
        total = cost_transcribe + cost_translate
        print(f"  Translation:       ${cost_translate:.4f}  "
              f"({tok_in:,} in / {tok_out:,} out tokens)")
        print(f"  TOTAL COST:        ${total:.4f}")
    else:
        print(f"  Translation:       unknown pricing  "
              f"({tok_in:,} in / {tok_out:,} out tokens)")
        print(f"  TOTAL COST:        ${cost_transcribe:.4f} + translation")
    print("═" * 60)


if __name__ == "__main__":
    main()
