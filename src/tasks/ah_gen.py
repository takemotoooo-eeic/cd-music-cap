import os
import re
from collections import defaultdict
from typing import Sequence

import datasets
import json
import librosa
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset
from tqdm import tqdm

from src import Define
from .utils import LLMJudgeWrapper, LOCAL_JUDGE_MODEL

# Official generative prompts from:
# https://github.com/kuan2jiu99/audio-hallucination/blob/main/interspeech2024/generative_tasks/inference.py
GENERATIVE_PROMPTS = [
    "Describe the audio.",
    "What do you hear?",
    "What can be inferred from the audio?",
    "This is a sound of",
    "Generate audio caption:",
]

HF_REPO = "kuanhuggingface/audiocaps_hallucination"
GENERIC_NOUNS = frozenset({
    "audio", "sound", "sounds", "clip", "recording", "file",
    "something", "thing", "things",
})

# Paper Sec. 2.3 / Limitations: GPT-4 judges hallucinated vs supported objects
# from the LALM caption plus ground-truth captions and labels.
AH_GEN_JUDGE_SYSTEM = (
    "You evaluate object hallucination in audio captions. "
    "Return JSON only."
)

AH_GEN_JUDGE_PROMPT = """You will judge object hallucination in a generated audio caption.

Use the ground-truth labels and ground-truth captions as the reference for which sound-producing objects are actually in the audio.

Ground-truth labels:
{labels}

Ground-truth captions:
{captions}

Generated caption:
{prediction}

Instructions:
1. List sound-producing objects / events that are actually present in the audio (gt_objects). Start from the labels, and add objects clearly mentioned in the ground-truth captions. Drop generic words such as audio, sound, clip, recording, background, distance.
2. List sound-producing objects / events mentioned in the generated caption (pred_objects).
3. If a generated object refers to the same thing as a ground-truth object, copy the ground-truth name. Match synonyms, paraphrases, and different parts of speech (examples: "people talking" or "a man speaking" -> "speech"; "dog barking" -> "dog" or "bow-wow" if that is the label; "machine stitching" -> "sewing machine").
4. List pred_objects that are NOT supported by the ground truth (hallucinated_objects).

Return JSON only, with this schema:
{{"gt_objects": ["..."], "pred_objects": ["..."], "hallucinated_objects": ["..."]}}
"""


def _normalize_objects(items) -> set[str]:
    if items is None:
        return set()
    if isinstance(items, str):
        raw = [x.strip() for x in re.split(r"[,;\n]", items) if x.strip()]
    elif isinstance(items, (list, tuple, set)):
        raw = [str(x).strip() for x in items if str(x).strip()]
    else:
        return set()
    objs: set[str] = set()
    for item in raw:
        lemma = item.lower().strip().strip("\"'`")
        if lemma and lemma not in GENERIC_NOUNS:
            objs.add(lemma)
    return objs


def _parse_judge_json(text: str) -> dict:
    if not (text or "").strip():
        return {}
    blob = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, flags=re.DOTALL)
    if fenced:
        blob = fenced.group(1)
    else:
        start, end = blob.find("{"), blob.rfind("}")
        if start != -1 and end != -1 and end > start:
            blob = blob[start:end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def llm_judge_objects(
    pred_text: str,
    labels: Sequence[str],
    captions: Sequence[str],
    llm: LLMJudgeWrapper,
) -> dict:
    """Paper GPT-4 evaluator: decompose hallucinated vs supported objects."""
    labels_text = ", ".join(str(x) for x in (labels or []) if str(x).strip()) or "(none)"
    captions_text = "\n".join(
        f"- {c}" for c in (captions or []) if str(c).strip()
    ) or "- (none)"
    user_prompt = AH_GEN_JUDGE_PROMPT.format(
        labels=labels_text,
        captions=captions_text,
        prediction=(pred_text or "").strip() or "(empty)",
    )
    print("user_prompt: ", user_prompt)
    raw = llm.generate(
        user_prompt,
        system_prompt=AH_GEN_JUDGE_SYSTEM,
        max_new_tokens=512,
    )
    parsed = _parse_judge_json(raw)
    print("parsed: ", parsed)
    if not parsed:
        print(f"[Warning] ah-gen judge JSON parse failed: {str(raw)[:200]!r}...")
    gt_objects = _normalize_objects(parsed.get("gt_objects"))
    pred_objects = _normalize_objects(parsed.get("pred_objects"))
    hallucinated = _normalize_objects(parsed.get("hallucinated_objects")) & pred_objects
    if pred_objects and not hallucinated and "hallucinated_objects" not in parsed:
        hallucinated = pred_objects - gt_objects
    return {
        "gt_objects": gt_objects,
        "pred_objects": pred_objects,
        "hallucinated_objects": hallucinated,
        "supported_objects": pred_objects - hallucinated,
    }


def cal_chair_score(label_object_set: set[str], prediction_object_set: set[str]) -> float:
    """Official CHAIR / ECHO_I. Empty prediction -> -1."""
    if len(prediction_object_set) == 0:
        return -1.0
    return 1.0 - (
        len(prediction_object_set.intersection(label_object_set)) / len(prediction_object_set)
    )


def cal_hal_score(chair_score: float) -> float:
    """Official Hal / ECHO_S for one caption. Empty prediction -> -1."""
    if chair_score == -1:
        return -1.0
    return 1.0 if chair_score != 0 else 0.0


def compute_generative_metrics(judged: dict) -> dict:
    gt_objects = judged["gt_objects"]
    pred_objects = judged["pred_objects"]
    hallucinated = judged["hallucinated_objects"]
    supported = judged["supported_objects"]
    # CHAIR_g / Cover_g from GPT-4 decomposed object lists (paper Sec. 2.3).
    chair = cal_chair_score(supported, pred_objects)
    cover = 0.0 if not gt_objects else len(supported) / len(gt_objects)
    hal = cal_hal_score(chair)
    return {
        "chair": chair,
        "cover": cover,
        "hal": hal,
        "pred_objects": sorted(pred_objects),
        "gt_objects": sorted(gt_objects),
        "supported_objects": sorted(supported),
        "hallucinated_objects": sorted(hallucinated),
    }


def _pct(values: Sequence[float]) -> float:
    valid = [v for v in values if v != -1]
    if not valid:
        return 0.0
    return round((sum(valid) / len(valid)) * 100, 2)


def format_generative_metrics(metrics: dict) -> str:
    lines = [
        "{:<10} | {:<10} | {:<10}".format("CHAIR", "Cover", "Hal"),
        "{:<10.2f} | {:<10.2f} | {:<10.2f}".format(
            metrics["chair"] * 100, metrics["cover"] * 100, metrics["hal"] * 100
        ),
    ]
    per_prompt = metrics.get("per_prompt") or {}
    if per_prompt:
        lines.append("")
        lines.append("Per prompt:")
        for prompt, stat in per_prompt.items():
            lines.append(
                f"  {prompt}: CHAIR {stat['chair'] * 100:.2f} | "
                f"Cover {stat['cover'] * 100:.2f} | Hal {stat['hal'] * 100:.2f} "
                f"(n={stat['n']})"
            )
    return "\n".join(lines) + "\n"


class AudioHallucinationGen(object):
    def __init__(self):
        self.cache_dir = f"{Define.CACHE_DIR}/AudioHallucination-Gen"
        self.data_info_path = f"{self.cache_dir}/data_info.json"
        if not os.path.isfile(self.data_info_path):
            self.parse()
        with open(self.data_info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

    def _load_src_dataset(self):
        if Define.AUDIOHALLUCINATION_DIR:
            local_dir = os.path.join(Define.AUDIOHALLUCINATION_DIR, "audiocaps_hallucination")
            if os.path.isdir(local_dir):
                return datasets.load_dataset(local_dir, split="test")
        return datasets.load_dataset(HF_REPO, split="test")

    def parse(self):
        src_dataset = self._load_src_dataset()
        os.makedirs(f"{self.cache_dir}/wav", exist_ok=True)
        res = []
        for instance in tqdm(src_dataset, desc="AudioHallucination-Gen"):
            youtube_id = instance["youtube_id"]
            wav_name = f"{youtube_id}.wav"
            wav_path = f"{self.cache_dir}/wav/{wav_name}"
            if not os.path.isfile(wav_path):
                wav = librosa.resample(
                    instance["audio"]["array"],
                    orig_sr=instance["audio"]["sampling_rate"],
                    target_sr=16000,
                )
                wavfile.write(wav_path, 16000, (wav * 32767).astype(np.int16))
            labels = [str(x) for x in (instance.get("label") or [])]
            captions = [str(x) for x in (instance.get("caption") or [])]
            res.append({
                "id": youtube_id,
                "audio_input_path": wav_name,
                "labels": labels,
                "captions": captions,
            })
        with open(self.data_info_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.info)

    def get(self, idx) -> dict:
        instance = self.info[idx]
        audio_path = f"{self.cache_dir}/wav/{instance['audio_input_path']}"
        audio_input, _ = librosa.load(audio_path, sr=16000)
        return {
            **instance,
            "audio_input": audio_input,
            "audio_path": audio_path,
        }


class AudioHallucinationGenSequence(Dataset):
    """AudioCaps generative hallucination: 5 official prompts x each clip."""

    def __init__(self, judge_mode: str = "") -> None:
        self.llm = None
        if judge_mode == "api":
            self.llm = LLMJudgeWrapper(
                mode="api",
                model_name="gpt-4o-2024-11-20",
                api_key=Define.API_KEY,
            )
        elif judge_mode == "local":
            self.llm = LLMJudgeWrapper(
                mode="local",
                model_name=LOCAL_JUDGE_MODEL,
            )
        else:
            raise ValueError(
                "ah-gen requires an LLM judge. Use task name ah-gen-ja (GPT-4o) "
                "or ah-gen-jl (Gemma)."
            )
        self.corpus = AudioHallucinationGen()
        self.samples = []
        for audio_idx in range(len(self.corpus)):
            rec = self.corpus.info[audio_idx]
            for prompt in GENERATIVE_PROMPTS:
                self.samples.append((audio_idx, prompt, rec["id"]))
        self._metrics: dict[str, dict] = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_idx, prompt, youtube_id = self.samples[idx]
        sample = self.corpus.get(audio_idx)
        labels = sample["labels"]
        return {
            "id": f"{youtube_id}::{prompt}",
            "audio_input": sample["audio_input"],
            "text_input": prompt,
            "output": ", ".join(labels),
            "audio_path": sample["audio_path"],
            "type": prompt,
            "labels": labels,
            "captions": sample["captions"],
        }

    def eval(self, pred: str, gt: str, question: str = "", sample: dict = None) -> float:
        sample = sample or {}
        labels = sample.get("labels") or [x.strip() for x in (gt or "").split(",") if x.strip()]
        captions = sample.get("captions") or []
        judged = llm_judge_objects(pred, labels, captions, self.llm)
        metrics = compute_generative_metrics(judged)
        sid = sample.get("id") or f"{question}"
        self._metrics[sid] = metrics
        return metrics["chair"]

    def instance_metrics(
        self,
        ids: Sequence[str],
        scores: Sequence[float],
        types: Sequence[str] | None = None,
    ) -> dict:
        chairs, covers, hals = [], [], []
        per_prompt: dict[str, dict[str, list]] = defaultdict(
            lambda: {"chair": [], "cover": [], "hal": []}
        )
        for sid, typ in zip(ids, types or [""] * len(ids)):
            m = self._metrics.get(sid)
            if m is None:
                raise KeyError(f"No cached ah-gen metrics for id={sid}")
            chairs.append(m["chair"])
            covers.append(m["cover"])
            hals.append(m["hal"])
            if typ:
                per_prompt[typ]["chair"].append(m["chair"])
                per_prompt[typ]["cover"].append(m["cover"])
                per_prompt[typ]["hal"].append(m["hal"])

        prompt_stats = {}
        for prompt in GENERATIVE_PROMPTS:
            if prompt not in per_prompt:
                continue
            stat = per_prompt[prompt]
            prompt_stats[prompt] = {
                "chair": _pct(stat["chair"]) / 100.0,
                "cover": _pct(stat["cover"]) / 100.0,
                "hal": _pct(stat["hal"]) / 100.0,
                "n": len(stat["cover"]),
            }
        chair = _pct(chairs) / 100.0
        cover = _pct(covers) / 100.0
        hal = _pct(hals) / 100.0
        return {
            "score": cover,
            "chair": chair,
            "cover": cover,
            "hal": hal,
            "n": len(covers),
            "n_valid_chair": sum(1 for x in chairs if x != -1),
            "per_prompt": prompt_stats,
        }

    def format_results(
        self,
        ids: Sequence[str],
        scores: Sequence[float],
        types: Sequence[str] | None = None,
    ) -> str:
        return format_generative_metrics(self.instance_metrics(ids, scores, types))
