"""
Adaptive Vector Steering (AVS).

Training-free residual-stream intervention from
"Adaptive Vector Steering" (arXiv:2510.12851).

Steering vectors are the last-token residual contrast
    V^l = F(x_audio, x_q)^l - F(x_neg, x_q)^l
and are injected at every generation step with a layer-wise
strength that keeps sum_l λ^l = L · λ.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch


def parse_avs_config(model_config: dict) -> dict:
    """Read AVS hyperparameters from `avs` (fallback: `aad`)."""
    cfg = model_config.get("avs")
    if cfg is None:
        cfg = model_config.get("aad", {})
    if not isinstance(cfg, dict):
        raise ValueError("avs/aad config must be a mapping")

    layers = cfg.get("layers", [13, 26])
    if not isinstance(layers, (list, tuple)) or len(layers) != 2:
        raise ValueError("avs.layers must be [start, end] (inclusive, 0-indexed)")
    li_start, li_end = int(layers[0]), int(layers[1])
    if li_start > li_end:
        raise ValueError(f"avs.layers start ({li_start}) > end ({li_end})")

    negative = cfg.get("negative", "silence")
    if negative not in ("none", "silence"):
        raise ValueError(f"avs.negative must be 'none' or 'silence', got {negative!r}")

    return {
        "negative": negative,
        "beta": float(cfg.get("beta", 0.5)),
        "lambda": float(cfg.get("lambda", 0.05)),
        "li_start": li_start,
        "li_end": li_end,
        # Paper §3.2.2 puts the last two layers in ld. That is already encoded
        # by the chosen `layers` range; set attenuate_last>0 only to force it.
        "attenuate_last": int(cfg.get("attenuate_last", 0)),
    }


def compute_adaptive_lambdas(
    num_layers: int,
    li_start: int,
    li_end: int,
    base_lambda: float,
    beta: float,
    attenuate_last: int = 0,
) -> tuple[list[float], set[int], set[int]]:
    """Per-layer strengths from eq. (4).

    λ^l = (1 + |ld|/|li| · β) λ   if l ∈ li
    λ^l = (1 - β) λ               if l ∈ ld

    Layers in `[li_start, li_end]` are increased; the rest (ld) are decreased
    so that Σ_l λ^l = L · λ. If `attenuate_last` > 0, the final layers are
    forced into ld (paper §3.2.2).
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    last_attenuated = set(range(max(0, num_layers - max(attenuate_last, 0)), num_layers))
    li = {l for l in range(li_start, li_end + 1) if 0 <= l < num_layers} - last_attenuated
    if not li:
        raise ValueError(
            f"li is empty (layers=[{li_start}, {li_end}], "
            f"num_layers={num_layers}, attenuate_last={attenuate_last})"
        )
    ld = set(range(num_layers)) - li
    n_li, n_ld = len(li), len(ld)

    lambdas = []
    for l in range(num_layers):
        if l in li:
            lambdas.append((1.0 + (n_ld / n_li) * beta) * base_lambda)
        else:
            lambdas.append((1.0 - beta) * base_lambda)
    return lambdas, li, ld


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Locate the LM decoder stack across Qwen / DeSTA / AF3 layouts."""
    paths = (
        ("model", "layers"),
        ("llm_model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "layers"),
    )
    for path in paths:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return obj
    raise AttributeError(f"Cannot find decoder layers on {type(model).__name__}")


def _last_token_index(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    batch, seq_len = hidden.shape[:2]
    if attention_mask is None:
        return torch.full((batch,), seq_len - 1, device=hidden.device, dtype=torch.long)
    mask = attention_mask
    if mask.dim() > 2:
        mask = mask.reshape(mask.shape[0], -1)
        mask = mask[:, -seq_len:] if mask.shape[-1] >= seq_len else mask
    return mask.to(device=hidden.device, dtype=torch.long).sum(dim=-1) - 1


def gather_last_token(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """hidden: (B, T, D) → (B, D) last non-padding token."""
    idx = _last_token_index(hidden, attention_mask)
    batch = hidden.shape[0]
    return hidden[torch.arange(batch, device=hidden.device), idx]


def layer_hidden_states(outputs, num_layers: int) -> Sequence[torch.Tensor]:
    hs = getattr(outputs, "hidden_states", None)
    if hs is None:
        raise RuntimeError("Model forward did not return hidden_states; pass output_hidden_states=True")
    if len(hs) >= num_layers + 1:
        return hs[1 : num_layers + 1]
    if len(hs) >= num_layers:
        return hs[-num_layers:]
    raise RuntimeError(f"Expected >= {num_layers} hidden states, got {len(hs)}")


def extract_steering_vectors(
    outputs_pos,
    outputs_neg,
    num_layers: int,
    attention_mask_pos: torch.Tensor | None,
    attention_mask_neg: torch.Tensor | None,
) -> list[torch.Tensor]:
    """V^l = h_pos^l[T] - h_neg^l[T] for each decoder layer."""
    pos_layers = layer_hidden_states(outputs_pos, num_layers)
    neg_layers = layer_hidden_states(outputs_neg, num_layers)
    if len(pos_layers) != len(neg_layers):
        raise RuntimeError(
            f"Positive/negative hidden-state counts differ: {len(pos_layers)} vs {len(neg_layers)}"
        )
    vectors = []
    for h_pos, h_neg in zip(pos_layers, neg_layers):
        v = gather_last_token(h_pos, attention_mask_pos) - gather_last_token(h_neg, attention_mask_neg)
        vectors.append(v.detach())
    return vectors


def _split_layer_output(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _rebuild_layer_output(hidden, rest):
    if rest is None:
        return hidden
    return (hidden,) + rest


def _steer_and_renorm(hidden: torch.Tensor, steer_vec: torch.Tensor, lam: float) -> torch.Tensor:
    """h̃ = (h + λv) · ||h|| / ||h + λv||  (eqs. 2–3)."""
    orig_norm = torch.linalg.vector_norm(hidden, dim=-1, keepdim=True)
    steered = hidden + lam * steer_vec.to(device=hidden.device, dtype=hidden.dtype)
    new_norm = torch.linalg.vector_norm(steered, dim=-1, keepdim=True).clamp_min(1e-8)
    return steered * (orig_norm / new_norm)


def make_steering_hook(steer_vec: torch.Tensor, lam: float) -> Callable:
    """Forward hook: add λv to the last-token residual and restore L2 norm."""

    def hook(module, inputs, output):
        hidden, rest = _split_layer_output(output)
        if hidden is None or hidden.dim() < 2:
            return output
        last = hidden[:, -1:, :]
        last = _steer_and_renorm(last, steer_vec.view(1, 1, -1), lam)
        hidden = hidden.clone()
        hidden[:, -1:, :] = last
        return _rebuild_layer_output(hidden, rest)

    return hook


class DenseHiddenStateController:
    """Compute per-sample steering vectors and inject them during generate()."""

    def __init__(self, model: torch.nn.Module, model_config: dict):
        self.model = model
        self.cfg = parse_avs_config(model_config)
        self.layers = get_decoder_layers(model)
        self.num_layers = len(self.layers)
        self.lambdas, self.li, self.ld = compute_adaptive_lambdas(
            num_layers=self.num_layers,
            li_start=self.cfg["li_start"],
            li_end=self.cfg["li_end"],
            base_lambda=self.cfg["lambda"],
            beta=self.cfg["beta"],
            attenuate_last=self.cfg["attenuate_last"],
        )
        self.hooks: list = []
        li_sorted = sorted(self.li)
        ld_lambda = self.lambdas[next(iter(self.ld))] if self.ld else 0.0
        print(
            f"AVS: L={self.num_layers} li=[{li_sorted[0]}..{li_sorted[-1]}] "
            f"(|li|={len(self.li)}, |ld|={len(self.ld)}) "
            f"β={self.cfg['beta']} λ={self.cfg['lambda']} "
            f"λ_li={self.lambdas[li_sorted[0]]:.4f} λ_ld={ld_lambda:.4f} "
            f"negative={self.cfg['negative']}"
        )

    @property
    def negative(self) -> str:
        return self.cfg["negative"]

    def compute_steering_vectors(
        self,
        inputs_pos: dict,
        inputs_neg: dict,
        forward_fn: Callable | None = None,
    ) -> list[torch.Tensor]:
        if forward_fn is None:
            forward_fn = self._default_forward
        last_pos = self._last_tokens(forward_fn(self.model, inputs_pos), inputs_pos.get("attention_mask"))
        last_neg = self._last_tokens(forward_fn(self.model, inputs_neg), inputs_neg.get("attention_mask"))
        return [(hp - hn).detach() for hp, hn in zip(last_pos, last_neg)]

    def _last_tokens(self, outputs, attention_mask: torch.Tensor | None) -> list[torch.Tensor]:
        tokens = [
            gather_last_token(h, attention_mask).detach()
            for h in layer_hidden_states(outputs, self.num_layers)
        ]
        del outputs
        return tokens

    def register(self, steering_vectors: Iterable[torch.Tensor]) -> None:
        self.remove()
        vectors = list(steering_vectors)
        if len(vectors) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} steering vectors, got {len(vectors)}")
        for layer, vec, lam in zip(self.layers, vectors, self.lambdas):
            if lam == 0.0:
                continue
            self.hooks.append(layer.register_forward_hook(make_steering_hook(vec, lam)))

    def remove(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    @staticmethod
    def _default_forward(model, inputs):
        return model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
