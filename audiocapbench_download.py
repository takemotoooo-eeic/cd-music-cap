#!/usr/bin/env python3
"""Download the official AudioCapBench 1,000-sample eval set.

Faithful port of SalesforceAIResearch/AudioCapBench `audiocapbench/build_dataset.py`.
Uses the curated IDs in `data/audiocapbench/eval_data_ids/` and writes
`metadata.json` plus WAV files in the same layout as the original repo.

Usage:
    uv run python audiocapbench_download.py
    uv run python audiocapbench_download.py --dry-run
    uv run python audiocapbench_download.py --sound-clotho 2 --sound-audiocaps 2 --music 2 --speech 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

SEED = 42
CLOTHO_HF_REPO = "piyushsinghpasi/clotho-multilingual"
AUDIOCAPS_HF_REPO = "OpenSound/AudioCaps"
MUSICCAPS_AUDIO_HF_REPO = "kelvincai/MusicCaps_30s_wav"
MUSICCAPS_META_HF_REPO = "google/MusicCaps"
SPEECH_HF_REPO = "seastar105/emo_speech_caption_test"

_IDS_DIR = _ROOT / "data" / "audiocapbench" / "eval_data_ids"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")


def _write_wav(audio_array: np.ndarray, sr: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio_array, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype("<i2")
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def _load_csv_ids(filename: str, key_column: str) -> Optional[List[str]]:
    rows = _load_csv_rows(filename)
    if not rows:
        return None
    ids = [(row.get(key_column) or "").strip() for row in rows]
    ids = [v for v in ids if v]
    return ids or None


def _load_csv_rows(filename: str) -> Optional[List[dict]]:
    path = _IDS_DIR / filename
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows or None


def _captions_from_musiccaps(caption: str, aspect_list: str) -> List[str]:
    captions = [caption] if caption else []
    if aspect_list:
        try:
            aspects = eval(aspect_list) if str(aspect_list).startswith("[") else aspect_list
            if isinstance(aspects, list):
                aspect_text = ", ".join(str(a) for a in aspects)
            else:
                aspect_text = str(aspects)
        except Exception:
            aspect_text = str(aspect_list)
        captions.append(aspect_text)
    return captions or [""]


def _select_n(pool: List[str], n: int, seed: int) -> List[str]:
    if n >= len(pool):
        return list(pool)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(pool), size=n, replace=False)
    indices.sort()
    return [pool[i] for i in indices]


def _hf_kwargs(cache_dir: Optional[str], streaming: bool = False) -> dict:
    kwargs: dict = {}
    if streaming:
        kwargs["streaming"] = True
    # HF_HOME is the hub cache root; datasets already uses $HF_HOME/datasets.
    # Passing HF_HOME itself as cache_dir confuses the audio-folder builder.
    hf_home = os.environ.get("HF_HOME", "")
    if cache_dir:
        if not hf_home or os.path.abspath(cache_dir) != os.path.abspath(hf_home):
            kwargs["cache_dir"] = cache_dir
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


def _hf_snapshot(repo: str, retries: int = 5) -> Path:
    """Download a dataset repo (including Git LFS files) via huggingface_hub."""
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"    snapshot_download {repo} ...")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            path = snapshot_download(repo_id=repo, repo_type="dataset", token=token)
            return Path(path)
        except Exception as e:
            last_err = e
            wait = min(60, 5 * attempt)
            print(f"    snapshot_download failed ({e}). retry {attempt}/{retries} in {wait}s ...")
            time.sleep(wait)
    raise last_err


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio, int(sr)


def _load_hf_dataset(
    repo: str,
    split: str,
    cache_dir: Optional[str] = None,
    streaming: bool = True,
):
    """Load a HuggingFace dataset, falling back if streaming cannot parse metadata.

    Folder-based AudioFolder datasets (e.g. clotho-multilingual) store metadata.csv
    in Git LFS. datasets streaming then reads the LFS pointer instead of the CSV,
    which raises: `file_name` must be present in metadata files.
    """
    from datasets import load_dataset

    kwargs = _hf_kwargs(cache_dir, streaming=streaming)
    try:
        return load_dataset(repo, split=split, **kwargs)
    except ValueError as e:
        if not streaming:
            raise
        print(f"    streaming load failed for {repo}: {e}")
        print("    retrying without streaming (downloads LFS metadata/audio) ...")
        kwargs = _hf_kwargs(cache_dir, streaming=False)
        return load_dataset(repo, split=split, **kwargs)


def select_clotho_samples(n: int = 10, cache_dir: Optional[str] = None) -> List[dict]:
    """Load Clotho from the HF snapshot.

    `load_dataset(..., AudioFolder)` fails on datasets==3.6 with recent PyArrow:
    metadata `file_name` is inferred as large_string, not string.
    """
    import pandas as pd

    all_ids = _load_csv_ids("clotho_eval.csv", "audio_name")
    if not all_ids:
        all_ids = _load_csv_ids("clotho_ids.csv", "audio_name")

    root = _hf_snapshot(CLOTHO_HF_REPO)
    meta_path = root / "test" / "metadata.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Clotho metadata not found: {meta_path}")
    df = pd.read_csv(meta_path, usecols=["file_name", "audio_name", "caption"])
    df["audio_name"] = df["audio_name"].astype(str)
    df["file_name"] = df["file_name"].astype(str)

    if all_ids:
        chosen = all_ids[:n] if n < len(all_ids) else all_ids
        print(
            f"  Clotho: selected {len(chosen)} from {len(all_ids)} available, "
            f"reading from {root} ..."
        )
    else:
        chosen = _select_n(sorted(df["audio_name"].unique().tolist()), n, SEED)
        print(f"  Clotho: no ID file, selected {len(chosen)} from snapshot ...")

    target_set = set(chosen)
    grouped: Dict[str, dict] = {}
    for row in df.itertuples(index=False):
        name = row.audio_name
        if name not in target_set:
            continue
        if name not in grouped:
            grouped[name] = {"file_name": row.file_name, "captions": []}
        caption = row.caption
        if isinstance(caption, str) and caption:
            grouped[name]["captions"].append(caption)

    selected = []
    for i, name in enumerate(chosen):
        if name not in grouped:
            print(f"    Warning: {name} not found in dataset")
            continue
        wav_path = root / "test" / grouped[name]["file_name"]
        if not wav_path.is_file():
            print(f"    Warning: missing audio {wav_path}")
            continue
        audio_arr, sr = _read_wav(wav_path)
        selected.append({
            "id": f"clotho_{i}",
            "audio_name": name,
            "audio_array": audio_arr,
            "sr": sr,
            "duration": len(audio_arr) / sr,
            "reference_captions": grouped[name]["captions"] or [""],
            "source": "clotho_v2_test",
        })

    print(f"  Clotho: {len(selected)} samples ready")
    return selected


def select_audiocaps_samples(n: int = 10, cache_dir: Optional[str] = None) -> List[dict]:
    all_ids = _load_csv_ids("audiocaps_eval.csv", "youtube_id")
    if not all_ids:
        all_ids = _load_csv_ids("audiocaps_ids.csv", "youtube_id")

    if all_ids:
        chosen = all_ids[:n] if n < len(all_ids) else all_ids
        target_set = set(chosen)
        print(
            f"  AudioCaps: selected {len(chosen)} from {len(all_ids)} available, "
            f"fetching from {AUDIOCAPS_HF_REPO} ..."
        )
        ds = _load_hf_dataset(AUDIOCAPS_HF_REPO, "test", cache_dir, streaming=True)
        clips: Dict[str, dict] = {}
        row_count = 0
        for row in ds:
            row_count += 1
            if row_count % 500 == 0:
                print(
                    f"    streamed {row_count} rows, found {len(clips)}/{len(target_set)} clips ..."
                )
            ytid = row.get("youtube_id") or row.get("file_name", "")
            if ytid not in target_set:
                continue
            if ytid not in clips:
                clips[ytid] = {
                    "youtube_id": ytid,
                    "audio": row["audio"],
                    "captions": [],
                }
            clips[ytid]["captions"].append(row["caption"])

        selected = []
        for i, ytid in enumerate(chosen):
            if ytid not in clips:
                print(f"    Warning: {ytid} not found in dataset")
                continue
            clip = clips[ytid]
            audio = clip["audio"]
            audio_arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            selected.append({
                "id": f"audiocaps_{i}",
                "youtube_id": ytid,
                "audio_array": audio_arr,
                "sr": sr,
                "duration": len(audio_arr) / sr,
                "reference_captions": clip["captions"],
                "source": "audiocaps_test",
            })
    else:
        print(f"  AudioCaps: no ID file, loading all from {AUDIOCAPS_HF_REPO} ...")
        ds = _load_hf_dataset(AUDIOCAPS_HF_REPO, "test", cache_dir, streaming=True)
        clips_all: Dict[str, dict] = OrderedDict()
        for row in ds:
            ytid = row.get("youtube_id") or row.get("file_name", "")
            if ytid not in clips_all:
                clips_all[ytid] = {
                    "youtube_id": ytid,
                    "audio": row["audio"],
                    "captions": [],
                }
            clips_all[ytid]["captions"].append(row["caption"])
        clip_list = sorted(clips_all.values(), key=lambda c: c["youtube_id"])
        chosen_ytids = _select_n([c["youtube_id"] for c in clip_list], n, SEED + 1)
        ytid_to_clip = {c["youtube_id"]: c for c in clip_list}
        selected = []
        for i, ytid in enumerate(chosen_ytids):
            clip = ytid_to_clip[ytid]
            audio = clip["audio"]
            audio_arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            selected.append({
                "id": f"audiocaps_{i}",
                "youtube_id": ytid,
                "audio_array": audio_arr,
                "sr": sr,
                "duration": len(audio_arr) / sr,
                "reference_captions": clip["captions"],
                "source": "audiocaps_test",
            })

    print(f"  AudioCaps: {len(selected)} samples ready")
    return selected


def select_musiccaps_samples(n: int = 15, cache_dir: Optional[str] = None) -> List[dict]:
    """Load MusicCaps from the HF snapshot.

    `load_dataset(..., AudioFolder)` fails on this repo: metadata `file_name`
    is inferred as large_string, not string. The 30s wavs are already in the
    huggingface hub cache after a previous download.
    """
    eval_rows = _load_csv_rows("musiccaps_eval.csv") or _load_csv_rows("musiccaps_ids.csv")
    by_ytid: Dict[str, dict] = {}
    if eval_rows:
        all_ids = []
        for row in eval_rows:
            ytid = (row.get("ytid") or "").strip()
            if not ytid:
                continue
            all_ids.append(ytid)
            by_ytid[ytid] = row
    else:
        print(f"  MusicCaps: no CSV file, loading from {MUSICCAPS_META_HF_REPO} ...")
        meta_ds = _load_hf_dataset(
            MUSICCAPS_META_HF_REPO, "train", cache_dir, streaming=False
        )
        all_ids = sorted(r["ytid"] for r in meta_ds if r.get("is_audioset_eval"))

    chosen = all_ids[:n] if n < len(all_ids) else all_ids
    print(
        f"  MusicCaps: selected {len(chosen)} from {len(all_ids)} eval samples, "
        f"reading from snapshot {MUSICCAPS_AUDIO_HF_REPO} ..."
    )

    root = _hf_snapshot(MUSICCAPS_AUDIO_HF_REPO)
    meta_path = root / "metadata.csv"
    wav_meta: Dict[str, dict] = {}
    if meta_path.is_file():
        with open(meta_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ytid = (row.get("ytid") or "").strip()
                if ytid:
                    wav_meta[ytid] = row

    selected = []
    for j, ytid in enumerate(chosen):
        meta = by_ytid.get(ytid) or wav_meta.get(ytid) or {}
        file_name = (meta.get("file_name") or f"{ytid}.wav").strip()
        wav_path = root / file_name
        if not wav_path.is_file():
            wav_path = root / f"{ytid}.wav"
        if not wav_path.is_file():
            print(f"    Warning: {ytid} not found in snapshot ({wav_path})")
            continue
        audio_arr, sr = _read_wav(wav_path)

        caption = (meta.get("caption") or "").strip()
        aspect_list = meta.get("aspect_list") or ""
        captions = _captions_from_musiccaps(caption, aspect_list)

        try:
            start_s = float(meta.get("start_s") or 0)
            end_s = float(meta.get("end_s") or 0)
        except (TypeError, ValueError):
            start_s = end_s = 0.0
        if start_s and end_s and end_s > start_s:
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            if end_sample <= len(audio_arr):
                audio_arr = audio_arr[start_sample:end_sample]

        selected.append({
            "id": f"musiccaps_{j}",
            "ytid": ytid,
            "audio_array": audio_arr,
            "sr": sr,
            "duration": len(audio_arr) / sr,
            "reference_captions": captions,
            "aspect_list": aspect_list,
            "source": "musiccaps_eval",
        })

    print(f"  MusicCaps: {len(selected)} samples ready")
    return selected


def _speech_record(i: int, row: dict, transcript: str) -> dict:
    audio = row["audio"]
    caption = row.get("caption", "")
    audio_arr = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    refs = [caption] if caption else []
    if transcript:
        refs.append(f'{caption} The speaker says: "{transcript}"')
    if not refs:
        refs = [""]
    return {
        "id": f"speech_{i}",
        "audio_array": audio_arr,
        "sr": sr,
        "duration": len(audio_arr) / sr,
        "reference_captions": refs,
        "caption": caption,
        "transcript": transcript,
        "source": "emo_speech_caption_test",
    }


def _load_speech_dataset():
    """Load emo_speech_caption_test from a local snapshot (avoids HF API timeouts)."""
    from datasets import load_dataset

    root = _hf_snapshot(SPEECH_HF_REPO)
    data_dir = root / "data"
    return load_dataset("parquet", data_dir=str(data_dir), split="train")


def select_speech_samples(n: int = 15, cache_dir: Optional[str] = None) -> List[dict]:
    all_ids = _load_csv_ids("speech_eval.csv", "transcript")
    if not all_ids:
        all_ids = _load_csv_ids("speech_ids.csv", "transcript")

    print(
        f"  Speech: loading {SPEECH_HF_REPO} from snapshot ..."
    )
    ds = _load_speech_dataset()

    if all_ids:
        chosen = all_ids[:n] if n < len(all_ids) else all_ids
        target_set = set(chosen)
        print(f"  Speech: selected {len(chosen)} from {len(all_ids)} available")
        matched: Dict[str, dict] = {}
        for row_count, row in enumerate(ds, start=1):
            if row_count % 500 == 0:
                print(
                    f"    scanned {row_count} rows, found {len(matched)}/{len(target_set)} clips ..."
                )
            transcript = row.get("transcript", "")
            if transcript in target_set and transcript not in matched:
                matched[transcript] = row
                if len(matched) == len(target_set):
                    print(f"    all {len(target_set)} clips found after {row_count} rows")
                    break
        selected = []
        for i, transcript in enumerate(chosen):
            if transcript not in matched:
                print(f"    Warning: transcript not found: {transcript[:60]}...")
                continue
            selected.append(_speech_record(i, matched[transcript], transcript))
    else:
        transcripts_sorted = sorted(row.get("transcript", "") for row in ds)
        chosen_transcripts = _select_n(transcripts_sorted, n, SEED + 3)
        target_set = set(chosen_transcripts)
        matched = {}
        for row in ds:
            t = row.get("transcript", "")
            if t in target_set and t not in matched:
                matched[t] = row
                if len(matched) == len(target_set):
                    break
        selected = []
        for i, transcript in enumerate(chosen_transcripts):
            if transcript not in matched:
                continue
            selected.append(_speech_record(i, matched[transcript], transcript))

    print(f"  Speech: {len(selected)} samples ready")
    return selected


def build_test_set(
    output_dir: str,
    cache_dir: Optional[str] = None,
    dry_run: bool = False,
    sound_clotho_count: int = 200,
    sound_audiocaps_count: int = 200,
    music_count: int = 300,
    speech_count: int = 300,
) -> None:
    output_path = Path(output_dir)
    total = sound_clotho_count + sound_audiocaps_count + music_count + speech_count

    print("=" * 72)
    print("Building Audio Caption Test Set (AudioCapBench)")
    print("=" * 72)
    print(f"Output: {output_path}")
    print(
        f"Target: {total} samples "
        f"({sound_clotho_count} clotho + {sound_audiocaps_count} audiocaps "
        f"+ {music_count} music + {speech_count} speech)"
    )
    print(f"IDs: {_IDS_DIR}")
    print(f"Dry run: {dry_run}")
    print()

    if dry_run:
        for label, csv_name, key_col, n in [
            ("Clotho (sound)", "clotho_eval.csv", "audio_name", sound_clotho_count),
            ("AudioCaps (sound)", "audiocaps_eval.csv", "youtube_id", sound_audiocaps_count),
            ("MusicCaps (music)", "musiccaps_eval.csv", "ytid", music_count),
            ("Speech", "speech_eval.csv", "transcript", speech_count),
        ]:
            ids = _load_csv_ids(csv_name, key_col)
            if ids:
                chosen = ids[:n] if n < len(ids) else ids
                print(f"  {label}: {len(chosen)} samples from {csv_name} ({len(ids)} available)")
                for item in chosen[:3]:
                    print(f"    {item[:80]}...")
                if len(chosen) > 3:
                    print(f"    ... and {len(chosen) - 3} more")
            else:
                print(f"  {label}: CSV not found ({csv_name})")
        print()
        print("Dry run complete. No files downloaded or written.")
        return

    print("[1/4] Clotho v2 test samples (sound) ...")
    clotho_samples = select_clotho_samples(n=sound_clotho_count, cache_dir=cache_dir)
    print()
    print("[2/4] AudioCaps test samples (sound) ...")
    audiocaps_samples = select_audiocaps_samples(n=sound_audiocaps_count, cache_dir=cache_dir)
    print()
    print("[3/4] MusicCaps samples (music) ...")
    musiccaps_samples = select_musiccaps_samples(n=music_count, cache_dir=cache_dir)
    print()
    print("[4/4] Speech caption samples ...")
    speech_samples = select_speech_samples(n=speech_count, cache_dir=cache_dir)
    print()

    print("=" * 72)
    print("Writing audio files and metadata ...")
    print("=" * 72)

    sound_dir = output_path / "sound"
    music_dir = output_path / "music"
    speech_dir = output_path / "speech"
    sound_dir.mkdir(parents=True, exist_ok=True)
    music_dir.mkdir(parents=True, exist_ok=True)
    speech_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "description": "AudioCapBench - Audio Captioning Benchmark Test Set",
        "categories": {
            "sound": {
                "count": len(clotho_samples) + len(audiocaps_samples),
                "sources": ["clotho_v2_test", "audiocaps_test"],
            },
            "music": {"count": len(musiccaps_samples), "sources": ["musiccaps_eval"]},
            "speech": {"count": len(speech_samples), "sources": ["emo_speech_caption_test"]},
        },
        "total_samples": (
            len(clotho_samples) + len(audiocaps_samples)
            + len(musiccaps_samples) + len(speech_samples)
        ),
        "samples": [],
    }

    for sample in clotho_samples:
        fname = f"{sample['id']}.wav"
        wav_path = sound_dir / fname
        if not wav_path.is_file():
            _write_wav(sample["audio_array"], sample["sr"], wav_path)
        print(f"  Wrote {wav_path.name}")
        metadata["samples"].append({
            "id": sample["id"],
            "category": "sound",
            "source": sample["source"],
            "audio_file": f"sound/{fname}",
            "duration_s": round(sample["duration"], 2),
            "reference_captions": sample["reference_captions"],
        })

    for sample in audiocaps_samples:
        fname = f"{sample['id']}.wav"
        wav_path = sound_dir / fname
        if not wav_path.is_file():
            _write_wav(sample["audio_array"], sample["sr"], wav_path)
        print(f"  Wrote {wav_path.name}")
        metadata["samples"].append({
            "id": sample["id"],
            "category": "sound",
            "source": sample["source"],
            "youtube_id": sample["youtube_id"],
            "audio_file": f"sound/{fname}",
            "duration_s": round(sample["duration"], 2),
            "reference_captions": sample["reference_captions"],
        })

    for sample in musiccaps_samples:
        fname = f"{sample['id']}.wav"
        wav_path = music_dir / fname
        if not wav_path.is_file():
            _write_wav(sample["audio_array"], sample["sr"], wav_path)
        print(f"  Wrote {wav_path.name}")
        metadata["samples"].append({
            "id": sample["id"],
            "category": "music",
            "source": sample["source"],
            "ytid": sample["ytid"],
            "audio_file": f"music/{fname}",
            "duration_s": round(sample["duration"], 2),
            "reference_captions": sample["reference_captions"],
            "aspect_list": sample["aspect_list"],
        })

    for sample in speech_samples:
        fname = f"{sample['id']}.wav"
        wav_path = speech_dir / fname
        if not wav_path.is_file():
            _write_wav(sample["audio_array"], sample["sr"], wav_path)
        print(f"  Wrote {wav_path.name}")
        metadata["samples"].append({
            "id": sample["id"],
            "category": "speech",
            "source": sample["source"],
            "audio_file": f"speech/{fname}",
            "duration_s": round(sample["duration"], 2),
            "reference_captions": sample["reference_captions"],
            "caption": sample["caption"],
            "transcript": sample["transcript"],
        })

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved to: {metadata_path}")

    print()
    print("=" * 72)
    print("Audio Caption Test Set Summary")
    print("=" * 72)
    by_cat = {}
    for s in metadata["samples"]:
        by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count} samples")
    print(f"  Total: {len(metadata['samples'])} samples")
    print(f"\nOutput directory: {output_path}")
    print(f"Metadata: {metadata_path}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Build AudioCapBench test set (data downloaded from HuggingFace)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("AUDIOCAPBENCH_DIR", str(_ROOT / "data" / "audio_caption")),
        help="Output directory for the test set",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.getenv("HF_HOME", ""),
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sound-clotho", type=int, default=200)
    parser.add_argument("--sound-audiocaps", type=int, default=200)
    parser.add_argument("--music", type=int, default=300)
    parser.add_argument("--speech", type=int, default=300)
    parser.add_argument(
        "--ids-dir",
        type=str,
        default=None,
        help="Directory with CSV ID files (default: data/audiocapbench/eval_data_ids/)",
    )
    args = parser.parse_args()

    global _IDS_DIR
    if args.ids_dir:
        _IDS_DIR = Path(args.ids_dir)

    build_test_set(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir or None,
        dry_run=args.dry_run,
        sound_clotho_count=args.sound_clotho,
        sound_audiocaps_count=args.sound_audiocaps,
        music_count=args.music,
        speech_count=args.speech,
    )


if __name__ == "__main__":
    main()
