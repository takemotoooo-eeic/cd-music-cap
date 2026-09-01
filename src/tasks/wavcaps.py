"""WavCaps loader for SAE activation collection.

Default source is AudioSet Strongly-labelled subset (108,317 × 10 s clips).
Audio files are `{id}` with `.wav` replaced by `.flac` after unzipping
(https://huggingface.co/datasets/cvssp/WavCaps), e.g. `Yb0RFKhbpFJA.flac`.
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
from torch.utils.data import Dataset

from src import Define

AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg"}
SOURCE_JSON = {
    "FreeSound": ("json_files", "FreeSound", "fsd_final.json"),
    "BBC_Sound_Effects": ("json_files", "BBC_Sound_Effects", "bbc_final.json"),
    "SoundBible": ("json_files", "SoundBible", "sb_final.json"),
    "AudioSet_SL": ("json_files", "AudioSet_SL", "as_final.json"),
}
SOURCE_FLAC_DIRNAME = {
    "FreeSound": "FreeSound_flac",
    "BBC_Sound_Effects": "BBC_Sound_Effects_flac",
    "SoundBible": "SoundBible_flac",
    "AudioSet_SL": "AudioSet_SL_flac",
}


def wavcaps_root(explicit: str | Path | None = None) -> Path:
    root = Path(explicit) if explicit else Path(Define.WAVCAPS_DIR or "")
    if not root.is_dir():
        raise FileNotFoundError(
            "WavCaps root not found. Set WAVCAPS_DIR in .env "
            f"(current: {root!s})."
        )
    return root


def _dir_has_audio(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.rglob("*"):
        if child.is_file() and child.suffix.lower() in AUDIO_EXTS:
            return True
    return False


def default_audio_dirs(root: Path, source: str) -> list[Path]:
    flac_name = SOURCE_FLAC_DIRNAME.get(source, f"{source}_flac")
    nested = root / "mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms" / flac_name
    return [
        root / "audio" / source,
        root / "waveforms" / flac_name,
        nested,
        root / source,
    ]


def extract_hint(root: Path, source: str) -> str:
    zip_dir = root / "Zip_files" / source
    example = "Yb0RFKhbpFJA.flac" if source == "AudioSet_SL" else "180913.flac"
    return (
        f"No extracted {source} audio under {root}. Unzip the official archive, then "
        f"pass the flac directory with --audio-dir. Example:\n"
        f"  cd {zip_dir}\n"
        f"  zip -s 0 {source}.zip --out {source}_full.zip\n"
        f"  unzip {source}_full.zip -d {root}\n"
        f"Files are named like {example}."
    )


def _index_audio_dir(audio_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in audio_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            index[path.stem] = path
            index[path.name] = path
    return index


def _lookup_audio(audio_index: dict[str, Path], clip_id: str) -> Path | None:
    """Map WavCaps ids to files. AudioSet_SL ids look like `Yxxxxx.wav`."""
    keys = (clip_id, Path(clip_id).stem, clip_id.replace(".wav", "").replace(".WAV", ""))
    for key in keys:
        if key in audio_index:
            return audio_index[key]
    return None


def resolve_audio_dir(root: Path, source: str, audio_dir: str | Path | None = None) -> Path:
    if audio_dir is not None:
        path = Path(audio_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"--audio-dir is not a directory: {path}")
        return path
    for candidate in default_audio_dirs(root, source):
        if _dir_has_audio(candidate):
            return candidate
    raise FileNotFoundError(extract_hint(root, source))


def load_wavcaps_records(root: Path, source: str) -> list[dict]:
    if source not in SOURCE_JSON:
        raise ValueError(f"Unknown WavCaps source {source!r}; choose from {sorted(SOURCE_JSON)}")
    json_path = root.joinpath(*SOURCE_JSON[source])
    if not json_path.is_file():
        raise FileNotFoundError(f"WavCaps metadata not found: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not records:
        raise ValueError(f"No clips in {json_path}")
    return records


class WavCapsSequence(Dataset):
    """WavCaps clips for residual-stream collection (no eval)."""

    def __init__(
        self,
        source: str = "AudioSet_SL",
        prompt: str = "Generate a detailed caption for the audio clip.",
        root: str | Path | None = None,
        audio_dir: str | Path | None = None,
        max_duration: float | None = 30.0,
        sampling_rate: int = 16000,
    ) -> None:
        self.source = source
        self.prompt = prompt
        self.max_duration = max_duration
        self.sampling_rate = sampling_rate
        self.root = wavcaps_root(root)
        self.audio_dir = resolve_audio_dir(self.root, source, audio_dir)
        raw = load_wavcaps_records(self.root, source)
        audio_index = _index_audio_dir(self.audio_dir)
        self.records = []
        n_missing = 0
        for rec in raw:
            clip_id = str(rec["id"])
            path = _lookup_audio(audio_index, clip_id)
            if path is None:
                n_missing += 1
                continue
            self.records.append({**rec, "id": clip_id, "audio_path": str(path)})
        if not self.records:
            raise FileNotFoundError(
                f"Indexed 0/{len(raw)} {source} clips in {self.audio_dir}. "
                + extract_hint(self.root, source)
            )
        print(
            f"WavCaps {source}: {len(self.records)}/{len(raw)} clips with audio "
            f"({n_missing} missing) under {self.audio_dir}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav, _ = librosa.load(rec["audio_path"], sr=self.sampling_rate, mono=True)
        wav = np.asarray(wav, dtype=np.float32)
        if self.max_duration is not None and self.max_duration > 0:
            n_max = int(self.max_duration * self.sampling_rate)
            if wav.shape[0] > n_max:
                wav = wav[:n_max]
        duration = float(wav.shape[0] / self.sampling_rate)
        return {
            "id": rec["id"],
            "audio_input": wav,
            "text_input": self.prompt,
            "output": rec.get("caption", ""),
            "audio_path": rec["audio_path"],
            "duration": duration,
            "caption": rec.get("caption", ""),
        }
