#!/usr/bin/env python3
"""
jimaku — Generate Japanese learning subtitles from video/audio files.

For each subtitle segment produces three lines:
  1. Original Japanese text (kanji)
  2. Hiragana reading
  3. English translation (from existing English subtitles)

Uses ElevenLabs Scribe for Japanese speech transcription, guided by
English subtitles embedded in the video (or supplied separately).
An LLM then matches Japanese segments to English subtitles and adds
hiragana readings.

Requirements:
  - ffmpeg and ffprobe on PATH
  - OPENAI_API_KEY environment variable
  - ELEVENLABS_API_KEY environment variable
  - pip install openai elevenlabs

Usage:
  python jimaku.py movie.mkv
  python jimaku.py movie.mkv -o subs.srt
  python jimaku.py movie.mkv --english-srt movie.en.srt
"""

import argparse
from difflib import SequenceMatcher
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
    ElevenLabs = None

# ── Pricing (USD, Feb 2026) ───────────────────────────────────────────────
SCRIBE_COST_PER_MIN = 0.0067  # ElevenLabs Scribe

CHAT_PRICING = {                                       # per 1 M tokens
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
    "gpt-4.1":     {"input": 2.00,  "output": 8.00},
    "gpt-5":       {"input": 1.25,  "output": 10.00},
    "gpt-5.1":     {"input": 1.25,  "output": 10.00},
    "gpt-5.2":     {"input": 1.75,  "output": 14.00},
}

# ── Defaults ──────────────────────────────────────────────────────────────
MATCH_BATCH = 30           # English subs per matching call
MAX_RETRIES = 3
RETRY_DELAY = 5            # seconds


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def fatal(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def check_deps() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            fatal(f"{tool} not found on PATH. Install ffmpeg first.")
    if not os.environ.get("OPENAI_API_KEY"):
        fatal("OPENAI_API_KEY environment variable is not set.")
    if not os.environ.get("ELEVENLABS_API_KEY"):
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
    """Extract mono 16 kHz WAV from a video file.

    WAV is used instead of MP3 to avoid encoder-delay timing drift
    (~0.14 %/min with MP3, which accumulates to ~10 s over a 2-hour movie).
    """
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video,
         "-vn", "-ac", "1", "-ar", "16000",
         "-f", "wav", out],
        capture_output=True,
    )
    if r.returncode != 0:
        fatal(f"ffmpeg audio extraction failed:\n{r.stderr.decode()[-500:]}")




# ═══════════════════════════════════════════════════════════════════════════
#  Subtitle-guided chunking
# ═══════════════════════════════════════════════════════════════════════════

# Padding (seconds) around English subtitle clusters when extracting audio
_GUIDE_PAD_BEFORE = 2.0   # before the first sub in a cluster
_GUIDE_PAD_AFTER  = 3.0   # after the last sub in a cluster
_GUIDE_GAP        = 8.0   # merge subs closer than this into one cluster


def subtitle_guided_fragments(
    audio: str,
    en_subs: list[dict],
    chunk_dir: str,
) -> list[tuple[str, float]]:
    """Extract short audio fragments around English subtitle clusters.

    Returns [(chunk_path, chunk_start_seconds), …] — same format as
    split_audio so existing transcribe functions work unchanged.
    """
    total_dur = probe_duration(audio)

    # 1. Cluster nearby English subs: merge subs within _GUIDE_GAP of each other
    clusters: list[tuple[float, float]] = []  # (start, end) of each cluster
    for sub in sorted(en_subs, key=lambda s: s["start"]):
        s, e = sub["start"], sub["end"]
        if clusters and s - clusters[-1][1] < _GUIDE_GAP:
            clusters[-1] = (clusters[-1][0], max(clusters[-1][1], e))
        else:
            clusters.append((s, e))

    # 2. Expand with padding and clamp to audio bounds
    fragments: list[tuple[float, float]] = []
    for cs, ce in clusters:
        fs = max(0.0, cs - _GUIDE_PAD_BEFORE)
        fe = min(total_dur, ce + _GUIDE_PAD_AFTER)
        # merge with previous fragment if overlapping
        if fragments and fs <= fragments[-1][1]:
            fragments[-1] = (fragments[-1][0], max(fragments[-1][1], fe))
        else:
            fragments.append((fs, fe))

    # 3. Extract each fragment as a WAV file
    chunks: list[tuple[str, float]] = []
    for i, (start, end) in enumerate(fragments):
        out = os.path.join(chunk_dir, f"frag_{i:04d}.wav")
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




def _group_words(words: list[dict]) -> list[dict]:
    """Group word-level timestamps into subtitle-sized segments.

    Each word dict must have keys: text, start, end.
    Returns list of {start, end, text, words}.
    """
    segments: list[dict] = []
    buf: list[dict] = []
    JP_SENT_END = set("。！？!?")

    for w in words:
        text = w["text"]
        start = w["start"]
        end = w["end"]
        if not text.strip():
            continue

        # skip stray noise words (single char spanning >10 s)
        content = re.sub(r'[。！？!?、,.\s…]', '', text)
        if len(content) <= 1 and (end - start) > 10.0:
            continue

        # decide whether to flush current buffer before adding this word
        if buf:
            gap = start - buf[-1]["end"]
            duration = end - buf[0]["start"]
            prev_text = buf[-1]["text"]
            sentence_break = prev_text and prev_text[-1] in JP_SENT_END
            if gap > 5.0 or (gap > 0.7 and duration > 1.0) or duration > 12.0 or (sentence_break and gap > 0.15):
                seg_text = "".join(b["text"] for b in buf).strip()
                if seg_text:
                    segments.append({
                        "start": buf[0]["start"],
                        "end": buf[-1]["end"],
                        "text": seg_text,
                        "words": list(buf),
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
                "words": list(buf),
            })

    # Cap segment duration and enforce minimum display time
    for seg in segments:
        text_len = len(re.sub(r'[。！？!?、,.\s…]', '', seg["text"]))
        max_dur = max(5.0, text_len / 3.0 + 8.0)
        if seg["end"] - seg["start"] > max_dur:
            seg["end"] = seg["start"] + max_dur
        # Minimum display duration: 1.5 s
        if seg["end"] - seg["start"] < 1.5:
            seg["end"] = seg["start"] + 1.5

    # Filter echo/repeat artifacts
    def _core(t: str) -> str:
        return re.sub(r'[。！？!?、,.\s…\u3000]', '', t)

    filtered: list[dict] = []
    for seg in segments:
        sc = _core(seg["text"])
        if not sc:
            continue
        is_echo = False
        for prev in filtered:
            if abs(seg["start"] - prev["start"]) > 180:
                continue
            pc = _core(prev["text"])
            if len(sc) >= 2 and (sc in pc or pc in sc):
                is_echo = True
                break
            if min(len(sc), len(pc)) >= 4:
                if SequenceMatcher(None, sc, pc).ratio() > 0.7:
                    is_echo = True
                    break
        if not is_echo:
            filtered.append(seg)
    return filtered


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

    # Extract words into a uniform format
    raw_words = resp.words or []
    words: list[dict] = []
    for w in raw_words:
        if getattr(w, "type", "word") != "word":
            continue
        text = w.text if isinstance(w, dict) else getattr(w, "text", "")
        start = w["start"] if isinstance(w, dict) else getattr(w, "start", 0)
        end = w["end"] if isinstance(w, dict) else getattr(w, "end", 0)
        words.append({"text": text, "start": start, "end": end})

    return _group_words(words)


def match_batch(
    client: OpenAI,
    ja_segments: list[dict],
    en_subs: list[dict],
    model: str,
    reasoning_effort: str | None = None,
    debug_dir: str | None = None,
    batch_label: str = "",
) -> tuple[list[dict], int, int]:
    """Match Japanese ASR text to English subtitles via LLM.

    The Japanese segments are combined into a single block of text and
    the LLM is asked to split/redistribute it to match each English
    subtitle.  This avoids dropping subs when Scribe grouped words
    differently than the English track.

    Returns (results, prompt_tokens, completion_tokens).
    Each result: {"en_index": int, "japanese": str, "kana": str}
    where en_index is the 1-based position in the *en_subs* list.
    """
    # Combine all Japanese segments with clause-level timestamps
    JP_CLAUSE_END = set("。！？!?")
    ja_lines = []
    for s in ja_segments:
        words = s.get("words", [])
        if not words:
            ja_lines.append(s["text"])
            continue
        # Merge character-level tokens into clauses split at
        # sentence-ending punctuation or gaps > 1s
        clauses: list[tuple[float, str]] = []  # (start_time, text)
        clause_start = words[0]["start"]
        clause_text = ""
        for i, w in enumerate(words):
            clause_text += w["text"]
            is_end = w["text"] and w["text"][-1] in JP_CLAUSE_END
            gap_after = (words[i + 1]["start"] - w["end"]
                         if i + 1 < len(words) else 999)
            if is_end or gap_after > 1.0:
                clauses.append((clause_start, clause_text))
                clause_text = ""
                if i + 1 < len(words):
                    clause_start = words[i + 1]["start"]
        if clause_text:
            clauses.append((clause_start, clause_text))
        ja_lines.append(" ".join(
            f"[{t:.1f}s]{txt}" for t, txt in clauses
        ))
    ja_text = "\n".join(ja_lines)

    en_lines = "\n".join(
        f"E{i+1}. [{s['start']:.1f}s] {s['text']}" for i, s in enumerate(en_subs)
    )

    system_prompt = (
        "You are a Japanese-language expert.\n"
        "The user provides:\n"
        "1) Japanese speech from automatic speech recognition, with "
        "word-level timestamps in seconds [Xs].\n"
        "2) Numbered English subtitles with timestamps — the "
        "professional translation of the same scene.\n\n"
        "Your task: for EVERY English subtitle, find and extract "
        "the corresponding Japanese words using TIMESTAMPS to "
        "align them. Words whose timestamps fall near an English "
        "subtitle's timestamp belong to that subtitle.\n\n"
        "• Fix recognition errors, wrong kanji, or garbled text "
        "using the English subtitle as context.\n"
        "• Provide the full hiragana reading of the corrected "
        "Japanese.\n\n"
        "Return JSON:\n"
        '{"results": [{"en_index": 1, '
        '"japanese": "<corrected Japanese>", '
        '"kana": "<hiragana reading>"}, ...]}\n\n'
        "Rules:\n"
        "• en_index: the E-number of the English subtitle.\n"
        "• Use TIMESTAMPS as the primary signal for matching. "
        "Japanese words at time T should match the English "
        "subtitle closest to time T.\n"
        "• Produce a result for EVERY English subtitle that "
        "corresponds to spoken dialogue. Only skip subtitles "
        "that describe non-speech sounds like "
        '"[music]", "[door closes]", etc.\n'
        "• IMPORTANT: use ONLY the Japanese text from the ASR "
        "block. Do NOT invent or fabricate Japanese that is not "
        "present in the ASR output. If the ASR block has no "
        "corresponding Japanese for an English subtitle, skip it.\n"
        "• You may fix minor ASR errors (wrong kanji, garbled "
        "characters) but the underlying words must come from the "
        "ASR block.\n"
        "• japanese: corrected Japanese text for that line.\n"
        "• kana: complete hiragana reading (convert all "
        "kanji/katakana).\n"
        "• Preserve the order of en_index.\n"
        "• Return ONLY valid JSON, nothing else."
    )

    user_content = (
        "--- JAPANESE SPEECH (ASR) ---\n" + ja_text
        + "\n\n--- ENGLISH SUBTITLES ---\n" + en_lines
    )

    def call():
        kwargs = dict(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["temperature"] = 0.3
        return client.chat.completions.create(**kwargs)

    resp = _retry(call, "Matching batch")
    raw_content = resp.choices[0].message.content
    data = json.loads(raw_content)
    results = data.get("results", [])

    if debug_dir:
        tag = batch_label or "batch"
        with open(os.path.join(debug_dir, f"{tag}_prompt_system.txt"), "w") as f:
            f.write(system_prompt)
        with open(os.path.join(debug_dir, f"{tag}_prompt_user.txt"), "w") as f:
            f.write(user_content)
        with open(os.path.join(debug_dir, f"{tag}_response.json"), "w") as f:
            json.dump({"raw": raw_content, "parsed": data,
                       "usage": {"prompt": resp.usage.prompt_tokens,
                                 "completion": resp.usage.completion_tokens}},
                      f, ensure_ascii=False, indent=2)
        with open(os.path.join(debug_dir, f"{tag}_ja_segments.json"), "w") as f:
            json.dump(ja_segments, f, ensure_ascii=False, indent=2)
        with open(os.path.join(debug_dir, f"{tag}_en_subs.json"), "w") as f:
            json.dump(en_subs, f, ensure_ascii=False, indent=2)

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
    ap.add_argument("--model", default="gpt-5.2",
                    help="LLM for matching/kana (default: gpt-5.2)")
    ap.add_argument("--english-srt",
                    help="Path to English .srt subtitles (default: "
                         "auto-extract from video)")
    ap.add_argument("--batch-size", type=int, default=MATCH_BATCH,
                    help="English subs per LLM call (default: 30)")
    ap.add_argument("--thinking", default="medium",
                    choices=["low", "medium", "high"],
                    help="Reasoning effort level (default: medium)")
    ap.add_argument("--debug-dir",
                    help="Save debug info (API inputs/outputs, segments) "
                         "to this directory")
    args = ap.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────
    video_path = args.video
    if not os.path.isfile(video_path):
        fatal(f"File not found: {video_path}")
    check_deps()

    out_path = args.output or str(Path(video_path).with_suffix(".srt"))
    client = OpenAI()
    debug_dir = args.debug_dir
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    cost_transcribe = 0.0
    cost_match = 0.0
    tok_in = 0
    tok_out = 0

    print(f"\n  jimaku — Japanese learning subtitle generator\n")

    with tempfile.TemporaryDirectory(prefix="jimaku_") as tmp:
        # ── Load English subtitles (required) ─────────────────────────────
        en_subs: list[dict] = []
        if args.english_srt:
            if not os.path.isfile(args.english_srt):
                fatal(f"English SRT not found: {args.english_srt}")
            en_subs = parse_srt(args.english_srt)
            print(f"  English subtitles: {len(en_subs)} loaded from "
                  f"{Path(args.english_srt).name}")
        else:
            embedded = extract_embedded_subs(video_path, tmp)
            if embedded:
                en_subs = parse_srt(embedded)
                print(f"  English subtitles: {len(en_subs)} extracted "
                      f"from video")
            else:
                fatal("No English subtitles found. Provide one with "
                      "--english-srt or embed in the video file.")

        # ── 1. Extract audio ──────────────────────────────────────────────
        audio = os.path.join(tmp, "audio.wav")
        print(f"[1/3] Extracting audio from {Path(video_path).name} …")
        t0 = time.time()
        extract_audio(video_path, audio)
        dur_s = probe_duration(audio)
        dur_m = dur_s / 60
        print(f"       Duration: {dur_m:.1f} min  |  "
              f"Extracted in {time.time() - t0:.1f}s")

        # ── 2. Transcribe (subtitle-guided Scribe) ───────────────────────
        chunk_dir = os.path.join(tmp, "guide_frags")
        os.makedirs(chunk_dir)
        frags = subtitle_guided_fragments(audio, en_subs, chunk_dir)
        frag_min = sum(probe_duration(p) for p, _ in frags) / 60

        print(f"[2/3] Transcribing (ElevenLabs Scribe) …")
        print(f"       {len(frags)} fragment(s) around English subs  "
              f"({frag_min:.1f} min of {dur_m:.1f} min)")

        all_segs: list[dict] = []
        t0 = time.time()
        for fi, (fpath, frag_start) in enumerate(frags):
            segs = _retry(
                lambda p=fpath: transcribe_scribe(
                    os.environ["ELEVENLABS_API_KEY"], p,
                ),
                f"Scribe fragment {fi + 1}",
            )
            for s in segs:
                s["start"] += frag_start
                s["end"]   += frag_start
                for w in s.get("words", []):
                    w["start"] += frag_start
                    w["end"]   += frag_start
            all_segs.extend(segs)
            if (fi + 1) % 10 == 0 or fi == len(frags) - 1:
                print(f"       Fragment {fi + 1}/{len(frags)}: "
                      f"{len(segs)} segments")

        cost_transcribe = frag_min * SCRIBE_COST_PER_MIN
        print(f"       ✓ {len(all_segs)} segments  "
              f"({time.time() - t0:.1f}s)  "
              f"cost: ${cost_transcribe:.4f}")

        if not all_segs:
            fatal("No speech segments detected.")

        if debug_dir:
            with open(os.path.join(debug_dir, "en_subs.json"), "w") as f:
                json.dump(en_subs, f, ensure_ascii=False, indent=2)
            with open(os.path.join(debug_dir, "scribe_segments.json"), "w") as f:
                json.dump(all_segs, f, ensure_ascii=False, indent=2)

        # ── 3. Match Japanese → English subs + add kana ───────────────────
        model = args.model
        reasoning_effort = args.thinking
        # Split into batches at large time gaps, with a max size cap.
        batch_size = args.batch_size
        batches: list[list[dict]] = []
        cur_batch: list[dict] = [en_subs[0]]
        for s in en_subs[1:]:
            gap = s["start"] - cur_batch[-1]["end"]
            if len(cur_batch) >= batch_size or gap > _GUIDE_GAP:
                batches.append(cur_batch)
                cur_batch = [s]
            else:
                cur_batch.append(s)
        batches.append(cur_batch)
        n_batches = len(batches)
        think_label = f" (thinking={reasoning_effort})" if reasoning_effort else ""
        print(f"[3/3] Matching with {model}{think_label}  "
              f"({n_batches} batch(es) of ≤{batch_size}) …")

        final_segs: list[dict] = []
        for bi, batch_en in enumerate(batches):
            t0 = time.time()

            # Time range for this English batch (with padding)
            t_start = batch_en[0]["start"] - _GUIDE_PAD_BEFORE - 5
            t_end = batch_en[-1]["end"] + _GUIDE_PAD_AFTER + 5
            batch_ja = [
                s for s in all_segs
                if s["start"] < t_end and s["end"] > t_start
            ]

            results, p_tok, c_tok = match_batch(
                client, batch_ja, batch_en, model,
                reasoning_effort=reasoning_effort,
                debug_dir=debug_dir,
                batch_label=f"batch_{bi+1:03d}",
            )
            tok_in += p_tok
            tok_out += c_tok
            batch_cost = translation_cost(model, p_tok, c_tok)
            if batch_cost is not None:
                cost_match += batch_cost

            # Build output segments using English sub timing
            for r in results:
                ei = r.get("en_index", 0) - 1   # 1-based → 0-based
                if not (0 <= ei < len(batch_en)):
                    continue
                en = batch_en[ei]
                final_segs.append({
                    "start":   en["start"],
                    "end":     en["end"],
                    "text":    r.get("japanese", ""),
                    "kana":    r.get("kana", ""),
                    "english": en["text"],
                })

            cost_str = (f"${cost_match:.4f}"
                        if cost_match > 0 else "unknown (see tokens)")
            print(f"       Batch {bi + 1}/{n_batches}: "
                  f"{len(batch_en)} en subs × {len(batch_ja)} ja segs  "
                  f"({time.time() - t0:.1f}s)  cost so far: {cost_str}")

        cost_str = (f"${cost_match:.4f}"
                    if cost_match > 0 else "unknown pricing")
        print(f"       ✓ Matching cost: {cost_str}")

        all_segs = final_segs

    # ── Write SRT ─────────────────────────────────────────────────────────
    write_srt(all_segs, out_path)

    print()
    print("═" * 60)
    print(f"  Output file:       {out_path}")
    print(f"  Subtitle entries:  {len(all_segs)}")
    print(f"  Audio duration:    {dur_m:.1f} min")
    print(f"  ─────────────────────────────────────")
    print(f"  Transcription:     ${cost_transcribe:.4f}")
    if cost_match > 0:
        total = cost_transcribe + cost_match
        print(f"  Matching:          ${cost_match:.4f}  "
              f"({tok_in:,} in / {tok_out:,} out tokens)")
        print(f"  TOTAL COST:        ${total:.4f}")
    else:
        print(f"  Matching:          unknown pricing  "
              f"({tok_in:,} in / {tok_out:,} out tokens)")
        print(f"  TOTAL COST:        ${cost_transcribe:.4f} + matching")
    print("═" * 60)


if __name__ == "__main__":
    main()
