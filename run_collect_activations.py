"""Collect LALM residual activations for SAE training (AR&D, arXiv:2602.22253).

Prefill-only forward on WavCaps AudioSet Strongly-labelled subset;
one float16 file per decoder layer.

  uv run python run_collect_activations.py -o wavcaps-audioset-sl -s qwen \
      --model_config config/activation_qwen.yaml
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import set_seed

load_dotenv(Path(__file__).resolve().parent / ".env")

from src.systems.generation.activation import (
    LayerActivationCollector,
    LayerActivationWriter,
    parse_activation_config,
)
from src.systems.load import load_system
from src.tasks.wavcaps import WavCapsSequence


def _sample_indices(n: int, start_index: int = 0, max_samples: int | None = None) -> range:
    start = max(0, start_index)
    if start >= n:
        raise ValueError(f"--start-index {start_index} is >= dataset size {n}")
    end = n if max_samples is None else min(start + max_samples, n)
    return range(start, end)


def _load_progress(output_dir: Path) -> dict:
    path = output_dir / "progress.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_progress(output_dir: Path, payload: dict) -> None:
    with open(output_dir / "progress.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def create_config(args) -> dict:
    res = {"system_name": args.system_name, "task_name": f"wavcaps-{args.source.lower()}"}
    res["model_config"] = {}
    for path in args.model_config:
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        res["model_config"].update(config)
    act = parse_activation_config(res["model_config"])
    if args.prompt is not None:
        act["prompt"] = args.prompt
    if args.max_duration is not None:
        act["max_duration"] = args.max_duration
    if args.pool is not None:
        act["pool"] = args.pool
    res["model_config"]["activation"] = act
    res["wavcaps"] = {
        "source": args.source,
        "audio_dir": args.audio_dir,
        "root": args.wavcaps_dir,
    }
    return res


def collect(args, config: dict) -> None:
    act = config["model_config"]["activation"]
    layers = act["layers"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    ds = WavCapsSequence(
        source=args.source,
        prompt=act["prompt"],
        root=args.wavcaps_dir,
        audio_dir=args.audio_dir,
        max_duration=act["max_duration"],
    )
    if args.dry_run:
        print(f"dry-run: {len(ds)} clips, layers={layers}, pool={act['pool']}")
        return

    progress = _load_progress(output_dir) if args.resume else {}
    resume_index = int(progress.get("next_index", args.start_index))
    start_index = max(args.start_index, resume_index)
    indices = _sample_indices(len(ds), start_index=start_index, max_samples=args.max_samples)
    print(
        f"Collecting layers={layers} pool={act['pool']} "
        f"max_duration={act['max_duration']} clips={len(indices)} "
        f"(index [{indices.start}:{indices.stop}]) → {output_dir}"
    )

    system = load_system(args.system_name, system_config=config, checkpoint=None)
    system.eval()
    system.cuda()
    collector = LayerActivationCollector(system.model, layers, pool=act["pool"])
    writer = (
        LayerActivationWriter.resume(output_dir, layers)
        if args.resume and (output_dir / f"layer_{layers[0]:02d}.json").is_file()
        else LayerActivationWriter(output_dir, layers)
    )
    index_path = output_dir / "index.jsonl"
    skip_path = output_dir / "skipped.jsonl"
    index_mode = "a" if args.resume and index_path.is_file() else "w"
    skip_mode = "a" if args.resume and skip_path.is_file() else "w"

    collector.register()
    n_ok = int(progress.get("n_ok", writer.n_clips))
    n_skip = int(progress.get("n_skip", 0))
    try:
        with open(index_path, index_mode, encoding="utf-8") as index_f, open(
            skip_path, skip_mode, encoding="utf-8"
        ) as skip_f:
            for i, idx in enumerate(tqdm(indices, total=len(indices))):
                sample = ds[idx]
                try:
                    acts = collector.collect(
                        system,
                        args.system_name,
                        [sample["audio_input"]],
                        [sample["text_input"]],
                    )
                    n_tokens = writer.append(acts)
                    rec = {
                        "id": sample["id"],
                        "dataset_index": idx,
                        "n_tokens": n_tokens,
                        "offset": writer.n_tokens - n_tokens,
                        "duration": sample.get("duration"),
                        "audio_path": sample["audio_path"],
                        "caption": sample.get("caption", ""),
                    }
                    index_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok += 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rec = {"id": sample["id"], "dataset_index": idx, "reason": "oom"}
                    skip_f.write(json.dumps(rec) + "\n")
                    n_skip += 1
                except Exception as exc:
                    rec = {
                        "id": sample["id"],
                        "dataset_index": idx,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    skip_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_skip += 1
                    if n_skip <= 5:
                        traceback.print_exc()
                if (i + 1) % args.flush_every == 0:
                    writer.flush()
                    index_f.flush()
                    skip_f.flush()
                    _save_progress(output_dir, {
                        "next_index": idx + 1,
                        "n_ok": n_ok,
                        "n_skip": n_skip,
                        "n_tokens": writer.n_tokens,
                    })
                del sample
                if (i + 1) % 8 == 0:
                    torch.cuda.empty_cache()
    finally:
        collector.remove()
        writer.close()
        _save_progress(output_dir, {
            "next_index": indices.stop,
            "n_ok": n_ok,
            "n_skip": n_skip,
            "n_tokens": writer.n_tokens,
            "done": True,
        })
        print(
            f"Saved {n_ok} clips ({writer.n_tokens} tokens) across layers {layers}; "
            f"skipped {n_skip}. Per-layer files under {output_dir}"
        )


def main(args):
    args.output_name = args.output_dir
    args.output_dir = f"activations/{args.system_name}/{args.output_dir}"
    config = create_config(args)
    collect(args, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect LALM residual activations for SAE training")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="run name under activations/<system>/")
    parser.add_argument("-s", "--system_name", type=str, default="qwen")
    parser.add_argument("--model_config", nargs="+", default=["config/activation_qwen.yaml"])
    parser.add_argument("--source", type=str, default="AudioSet_SL", help="WavCaps subset")
    parser.add_argument("--wavcaps-dir", type=str, default=None)
    parser.add_argument("--audio-dir", type=str, default=None, help="directory of {id}.flac files")
    parser.add_argument("--prompt", type=str, default=None, help="override activation.prompt")
    parser.add_argument("--max-duration", type=float, default=None, help="truncate audio (seconds)")
    parser.add_argument("--pool", type=str, choices=["none", "mean"], default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
