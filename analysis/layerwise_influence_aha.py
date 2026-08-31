"""
Layer-wise influence analysis (AVS paper §3.1 / Fig. 2) on AHa-Bench.

For each sample:
  V^l = h_audio^l[T] - h_silence^l[T]
After the sweep, the dataset-mean steering vector is the probe direction:
  sim_i^l = cosine(h_audio_i^l[T], mean_j V_j^l)
Samples are split by AHa-Bench correctness. The left panel plots mean cosine
similarity of correct vs incorrect groups; the right panel plots Cohen's d.

Usage:
  uv run python analysis/layerwise_influence_aha.py -o analysis_out/layerwise_aha_qwen -s qwen -t aha-jl --yes-no-only
  uv run python analysis/layerwise_influence_aha.py -o analysis_out/layerwise_aha_af3 -s af3 -t aha-jl --yes-no-only
  uv run python analysis/layerwise_influence_aha.py -o analysis_out/layerwise_aha_af3 --plot-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import torch

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import set_seed

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")

from src.systems.generation.avs import gather_last_token, get_decoder_layers, layer_hidden_states
from src.systems.generation.logits_process import _nested_omni_forward
from src.systems.load import load_system
from src.tasks.load import get_test_task


SYSTEM_NAMES = {
    "qwen": "Qwen2.5-Omni",
    "af3": "Audio Flamingo 3",
}


def _sample_indices(n: int, start_index: int = 0, max_samples: int | None = None) -> range:
    start = max(0, start_index)
    if start >= n:
        raise ValueError(f"--start-index {start_index} is >= dataset size {n}")
    end = n if max_samples is None else min(start + max_samples, n)
    return range(start, end)


def _qwen_forward(model, inputs):
    outputs, _ = _nested_omni_forward(
        model,
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return outputs


def _plain_forward(model, inputs):
    return model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )


def prepare_inputs(system, system_name: str, audios, texts, *, silence: bool):
    if system_name == "qwen":
        prompts = [system.format_prompt(t, audio_exist=True) for t in texts]
        audio_in = [np.zeros_like(a) for a in audios] if silence else audios
        return system.processor(
            audio=audio_in,
            text=prompts,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        ).to(system.device)

    if system_name == "af3":
        audio_in = np.zeros_like(audios[0]) if silence else audios[0]
        conversation = system.format_conversation(texts[0], audio_path_or_array=audio_in)
        return system.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(system.device)

    raise ValueError(f"Unsupported system for hidden-state probe: {system_name}")


def last_token_hidden_stack(system, system_name: str, audios, texts, *, silence: bool) -> np.ndarray:
    """(L, D) last-token residual of every decoder layer."""
    inputs = prepare_inputs(system, system_name, audios, texts, silence=silence)
    forward_fn = _qwen_forward if system_name == "qwen" else _plain_forward
    outputs = forward_fn(system.model, inputs)
    num_layers = len(get_decoder_layers(system.model))
    layers = layer_hidden_states(outputs, num_layers)
    mask = inputs.get("attention_mask")
    stack = torch.stack([gather_last_token(h, mask)[0] for h in layers], dim=0)
    vecs = stack.detach().float().cpu().numpy()
    del outputs, inputs, stack
    return vecs


def cohens_d(correct: np.ndarray, incorrect: np.ndarray) -> float:
    n_c, n_i = len(correct), len(incorrect)
    if n_c < 2 or n_i < 2:
        return float("nan")
    v_c = np.var(correct, ddof=1)
    v_i = np.var(incorrect, ddof=1)
    pooled = np.sqrt(((n_c - 1) * v_c + (n_i - 1) * v_i) / (n_c + n_i - 2))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(correct) - np.mean(incorrect)) / pooled)


def cosine_rows(hidden: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """hidden (N, L, D), direction (L, D) → (N, L)."""
    h = hidden / np.clip(np.linalg.norm(hidden, axis=-1, keepdims=True), 1e-8, None)
    d = direction / np.clip(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8, None)
    return np.einsum("nld,ld->nl", h, d)


def collect(args) -> dict:
    system = load_system(args.system, {"system_name": args.system, "model_config": {}})
    system.eval()
    system.cuda()
    ds = get_test_task(args.task)
    indices = _sample_indices(len(ds), start_index=args.start_index, max_samples=args.max_samples)

    hidden_pos, hidden_neg, records = [], [], []
    for idx in tqdm(indices, total=len(indices), desc=f"AHa-Bench ({args.system})"):
        sample = ds[idx]
        is_asr = "asr" in str(sample.get("type", "")).lower()
        if args.yes_no_only and is_asr:
            continue

        h_pos = last_token_hidden_stack(
            system, args.system, [sample["audio_input"]], [sample["text_input"]], silence=False
        )
        h_neg = last_token_hidden_stack(
            system, args.system, [sample["audio_input"]], [sample["text_input"]], silence=True
        )
        res = system.inference([sample["audio_input"]], [sample["text_input"]], [sample["audio_path"]])
        pred = res["prediction"]
        eval_kwargs = {}
        if "sample" in getattr(ds.eval, "__code__").co_varnames:
            eval_kwargs["sample"] = sample
        score = float(ds.eval(pred, sample["output"], sample["text_input"], **eval_kwargs))

        hidden_pos.append(h_pos)
        hidden_neg.append(h_neg)
        records.append({
            "id": sample["id"],
            "type": sample.get("type", ""),
            "prediction": pred,
            "gold": sample["output"],
            "score": score,
            "correct": score >= 0.5,
        })
        if args.save_every and len(records) % args.save_every == 0:
            _save_arrays(args.output_dir, hidden_pos, hidden_neg, records)

    payload = _save_arrays(args.output_dir, hidden_pos, hidden_neg, records)
    payload["system"] = args.system
    del system
    return payload


def _save_arrays(output_dir: str, hidden_pos, hidden_neg, records) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    pos = np.stack(hidden_pos, axis=0).astype(np.float32)
    neg = np.stack(hidden_neg, axis=0).astype(np.float32)
    np.savez_compressed(
        os.path.join(output_dir, "hidden_states.npz"),
        hidden_pos=pos,
        hidden_neg=neg,
    )
    with open(os.path.join(output_dir, "records.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"hidden_pos": pos, "hidden_neg": neg, "records": records}


def load_payload(output_dir: str) -> dict:
    data = np.load(os.path.join(output_dir, "hidden_states.npz"))
    records = []
    with open(os.path.join(output_dir, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    meta_path = os.path.join(output_dir, "run_meta.json")
    system_name = "qwen"
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            system_name = json.load(f).get("system", system_name)
    return {
        "hidden_pos": data["hidden_pos"],
        "hidden_neg": data["hidden_neg"],
        "records": records,
        "system": system_name,
    }


def analyze(payload: dict) -> dict:
    hidden_pos = payload["hidden_pos"]
    hidden_neg = payload["hidden_neg"]
    records = payload["records"]
    correct_mask = np.array([bool(r["correct"]) for r in records])
    if correct_mask.size == 0:
        raise RuntimeError("No samples collected")
    if not correct_mask.any() or not (~correct_mask).any():
        raise RuntimeError(
            f"Need both correct and incorrect samples "
            f"(correct={int(correct_mask.sum())}, incorrect={int((~correct_mask).sum())})"
        )

    steer = hidden_pos - hidden_neg
    direction = steer.mean(axis=0)
    sims = cosine_rows(hidden_pos, direction)

    n_layers = sims.shape[1]
    mean_correct = sims[correct_mask].mean(axis=0)
    mean_incorrect = sims[~correct_mask].mean(axis=0)
    effect = np.array([
        cohens_d(sims[correct_mask, l], sims[~correct_mask, l]) for l in range(n_layers)
    ])

    stats = {
        "n": int(len(records)),
        "n_correct": int(correct_mask.sum()),
        "n_incorrect": int((~correct_mask).sum()),
        "n_layers": n_layers,
        "mean_correct": mean_correct.tolist(),
        "mean_incorrect": mean_incorrect.tolist(),
        "cohens_d": effect.tolist(),
    }
    return stats


def plot_figure(stats: dict, output_path: str, model_name: str = "Qwen2.5-Omni") -> None:
    layers = np.arange(stats["n_layers"])
    mean_c = np.array(stats["mean_correct"])
    mean_i = np.array(stats["mean_incorrect"])
    effect = np.array(stats["cohens_d"])

    fig, (ax_sim, ax_d) = plt.subplots(1, 2, figsize=(14.5, 5.2))
    width = 0.38
    ax_sim.bar(layers - width / 2, mean_c, width, color="#2ca02c", label="Correct")
    ax_sim.bar(layers + width / 2, mean_i, width, color="#d62728", label="Incorrect")
    ax_sim.axhline(0.0, color="0.5", linewidth=0.8)
    ax_sim.set_xlabel("Layer")
    ax_sim.set_ylabel("Cosine Similarity")
    ax_sim.set_title("Cosine Similarity: Correct vs Incorrect")
    ax_sim.legend(frameon=True)
    ax_sim.set_xticks(layers[::2] if len(layers) > 16 else layers)
    ax_sim.grid(axis="y", linestyle=":", alpha=0.4)

    ax_d.bar(layers, effect, color="#c44e52", width=0.75)
    ax_d.axhline(0.2, color="#f4a261", linestyle="--", linewidth=1.2, label="Small effect")
    ax_d.axhline(0.5, color="#e9c46a", linestyle="--", linewidth=1.2, label="Medium effect")
    ax_d.axhline(0.8, color="#e76f51", linestyle="--", linewidth=1.2, label="Large effect")
    ax_d.axhline(0.0, color="0.5", linewidth=0.8)
    ax_d.set_xlabel("Layer")
    ax_d.set_ylabel("Cohen's d")
    ax_d.set_title("Effect Size by Layer")
    ax_d.legend(frameon=True, loc="upper left")
    ax_d.set_xticks(layers[::2] if len(layers) > 16 else layers)
    ax_d.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"Layer-wise Influence Analysis — {model_name} / AHa-Bench\n"
        f"(n={stats['n']}, correct={stats['n_correct']}, incorrect={stats['n_incorrect']})",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AVS §3.1 layer-wise influence on AHa-Bench")
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument(
        "-s", "--system", type=str, default="qwen", choices=sorted(SYSTEM_NAMES),
        help="model to probe",
    )
    parser.add_argument("-t", "--task", type=str, default="aha", help="aha / aha-jl / aha-ja")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--yes-no-only", action="store_true", help="skip ASR items")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--plot-only", action="store_true", help="replot from a previous run")
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot_only:
        payload = load_payload(args.output_dir)
        system_name = payload.get("system", args.system)
    else:
        payload = collect(args)
        system_name = args.system
        with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"system": system_name, "task": args.task}, f, indent=2)

    stats = analyze(payload)
    stats["system"] = system_name
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: stats[k] for k in ("n", "n_correct", "n_incorrect", "n_layers")}, indent=2))

    plot_figure(
        stats,
        os.path.join(args.output_dir, "layerwise_influence.png"),
        model_name=SYSTEM_NAMES.get(system_name, system_name),
    )


if __name__ == "__main__":
    main()
