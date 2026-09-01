"""Collect residual-stream activations for SAE training (AR&D / arXiv:2602.22253).

x ∈ R^{T × d} is the decoder-layer output (residual after block l). One binary
file is written per requested layer: float16 rows of shape (n_tokens, d).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .avs import _split_layer_output, get_decoder_layers
from .logits_process import _nested_omni_forward


def parse_activation_config(model_config: dict) -> dict:
    cfg = model_config.get("activation", {})
    if not isinstance(cfg, dict):
        raise ValueError("activation config must be a mapping")
    layers = cfg.get("layers")
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("activation.layers must be a non-empty list of 1-indexed layer ids")
    layers = [int(x) for x in layers]
    if any(i < 1 for i in layers):
        raise ValueError(f"activation.layers are 1-indexed; got {layers}")
    pool = str(cfg.get("pool", "none")).lower()
    if pool not in ("none", "mean"):
        raise ValueError(f"activation.pool must be 'none' or 'mean', got {pool!r}")
    max_duration = cfg.get("max_duration", 30.0)
    return {
        "layers": layers,
        "prompt": str(cfg.get("prompt", "Generate a detailed caption for the audio clip.")),
        "max_duration": None if max_duration in (None, "none") else float(max_duration),
        "pool": pool,
    }


def apply_token_mask(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """(B, T, D) + mask → (T', D) for batch size 1."""
    if hidden.dim() != 3:
        raise RuntimeError(f"Expected hidden (B, T, D), got {tuple(hidden.shape)}")
    if hidden.shape[0] != 1:
        raise RuntimeError("Activation collection currently supports batch size 1")
    tokens = hidden[0]
    if attention_mask is None:
        return tokens
    mask = attention_mask
    if mask.dim() > 2:
        mask = mask.reshape(mask.shape[0], -1)
        mask = mask[:, -tokens.shape[0]:] if mask.shape[-1] >= tokens.shape[0] else mask
    keep = mask[0].to(device=tokens.device) != 0
    if keep.numel() != tokens.shape[0]:
        n = min(keep.numel(), tokens.shape[0])
        return tokens[:n][keep[:n].bool()]
    return tokens[keep.bool()]


def _qwen_forward(model, inputs):
    outputs, _ = _nested_omni_forward(
        model,
        **inputs,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    return outputs


def _plain_forward(model, inputs):
    return model(
        **inputs,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )


def prepare_activation_inputs(system, system_name: str, audios, texts):
    if system_name.startswith("qwen"):
        prompts = [system.format_prompt(t, audio_exist=True) for t in texts]
        return system.processor(
            audio=audios,
            text=prompts,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        ).to(system.device)

    if system_name.startswith("af3"):
        conversation = system.format_conversation(texts[0], audio_path_or_array=audios[0])
        return system.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(system.device)

    raise ValueError(f"Unsupported system for activation collection: {system_name}")


class LayerActivationCollector:
    """Forward hooks on selected decoder layers; returns CPU float16 (T, D).

    `layers` are 1-indexed (layer 1 = first decoder block).
    """

    def __init__(self, model: torch.nn.Module, layers: Sequence[int], pool: str = "none"):
        self.model = model
        self.decoder_layers = get_decoder_layers(model)
        self.num_layers = len(self.decoder_layers)
        self.layer_ids = [int(i) for i in layers]
        for idx in self.layer_ids:
            if not 1 <= idx <= self.num_layers:
                raise ValueError(f"layer {idx} out of range [1, {self.num_layers}]")
        self.pool = pool
        self._cache: dict[int, torch.Tensor] = {}
        self._hooks = []

    def _hook(self, layer_idx: int):
        def fn(module, inputs, output):
            hidden, _ = _split_layer_output(output)
            self._cache[layer_idx] = hidden.detach()
        return fn

    def register(self) -> None:
        self.remove()
        for idx in self.layer_ids:
            self._hooks.append(self.decoder_layers[idx - 1].register_forward_hook(self._hook(idx)))

    def remove(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._cache = {}

    def collect(
        self,
        system,
        system_name: str,
        audios,
        texts,
    ) -> dict[int, np.ndarray]:
        self._cache = {}
        inputs = None
        outputs = None
        forward_fn = _qwen_forward if system_name.startswith("qwen") else _plain_forward
        try:
            inputs = prepare_activation_inputs(system, system_name, audios, texts)
            outputs = forward_fn(system.model, inputs)
            mask = inputs.get("attention_mask")
            out: dict[int, np.ndarray] = {}
            for idx in self.layer_ids:
                hidden = self._cache.get(idx)
                if hidden is None:
                    raise RuntimeError(f"Hook did not fire for layer {idx}")
                tokens = apply_token_mask(hidden, mask)
                if self.pool == "mean":
                    tokens = tokens.mean(dim=0, keepdim=True)
                arr = tokens.to(dtype=torch.float16).cpu().numpy()
                if arr.ndim != 2 or arr.shape[0] == 0:
                    raise RuntimeError(f"Empty activation at layer {idx}: shape={arr.shape}")
                out[idx] = np.ascontiguousarray(arr)
            return out
        finally:
            self._cache = {}
            del outputs, inputs


class LayerActivationWriter:
    """Append-only float16 matrix per layer: `layer_{idx:02d}.bin` + sidecar json."""

    def __init__(
        self,
        output_dir: str | Path,
        layer_ids: Sequence[int],
        hidden_size: int | None = None,
        mode: str = "wb",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layer_ids = [int(i) for i in layer_ids]
        self.hidden_size = hidden_size
        self.n_tokens = 0
        self.n_clips = 0
        self._files = {
            idx: open(self.bin_path(idx), mode)
            for idx in self.layer_ids
        }

    def bin_path(self, layer_idx: int) -> Path:
        return self.output_dir / f"layer_{layer_idx:02d}.bin"

    def meta_path(self, layer_idx: int) -> Path:
        return self.output_dir / f"layer_{layer_idx:02d}.json"

    def append(self, activations: dict[int, np.ndarray]) -> int:
        n = None
        for idx in self.layer_ids:
            arr = activations[idx]
            if arr.dtype != np.float16:
                arr = arr.astype(np.float16, copy=False)
            arr = np.ascontiguousarray(arr)
            if arr.ndim != 2:
                raise ValueError(f"layer {idx} expected (T, D), got {arr.shape}")
            if self.hidden_size is None:
                self.hidden_size = int(arr.shape[1])
            if arr.shape[1] != self.hidden_size:
                raise ValueError(f"hidden size mismatch: {arr.shape[1]} vs {self.hidden_size}")
            if n is None:
                n = int(arr.shape[0])
            elif arr.shape[0] != n:
                raise ValueError(f"token count mismatch across layers: {arr.shape[0]} vs {n}")
            self._files[idx].write(arr.tobytes(order="C"))
        assert n is not None
        self.n_tokens += n
        self.n_clips += 1
        return n

    def flush(self) -> None:
        for fh in self._files.values():
            fh.flush()
        self.write_meta()

    def write_meta(self) -> None:
        for idx in self.layer_ids:
            payload = {
                "layer": idx,
                "dtype": "float16",
                "hidden_size": self.hidden_size,
                "n_tokens": self.n_tokens,
                "n_clips": self.n_clips,
                "shape": [self.n_tokens, self.hidden_size],
                "file": self.bin_path(idx).name,
            }
            with open(self.meta_path(idx), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    def close(self) -> None:
        self.flush()
        for fh in self._files.values():
            fh.close()
        self._files = {}

    @classmethod
    def resume(cls, output_dir: str | Path, layer_ids: Sequence[int]) -> "LayerActivationWriter":
        output_dir = Path(output_dir)
        hidden_size = None
        n_tokens = 0
        n_clips = 0
        for idx in layer_ids:
            meta_path = output_dir / f"layer_{idx:02d}.json"
            if not meta_path.is_file():
                continue
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            hidden_size = meta.get("hidden_size", hidden_size)
            n_tokens = int(meta.get("n_tokens", 0))
            n_clips = int(meta.get("n_clips", 0))
            bin_path = output_dir / f"layer_{idx:02d}.bin"
            expected = n_tokens * int(hidden_size) * 2 if hidden_size else 0
            if bin_path.is_file() and expected:
                actual = bin_path.stat().st_size
                if actual < expected:
                    raise RuntimeError(
                        f"{bin_path} size {actual} < expected {expected} (corrupt resume state)"
                    )
                if actual > expected:
                    with open(bin_path, "r+b") as fh:
                        fh.truncate(expected)
        writer = cls(output_dir, layer_ids, hidden_size=hidden_size, mode="ab")
        writer.n_tokens = n_tokens
        writer.n_clips = n_clips
        return writer


def load_layer_activations(bin_path: str | Path) -> np.memmap:
    bin_path = Path(bin_path)
    meta_path = bin_path.with_suffix(".json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    dtype = np.dtype(meta["dtype"])
    shape = tuple(meta["shape"])
    return np.memmap(bin_path, dtype=dtype, mode="r", shape=shape)
