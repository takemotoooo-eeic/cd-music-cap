"""AudioCapBench: official captioning prompts and LLM-as-Judge metrics.

Protocol follows https://github.com/SalesforceAIResearch/AudioCapBench
(`evaluate.py` + `metrics.py`):
  - Category prompts rotated by sample index
  - LLM-as-Judge (Accuracy / Completeness / Hallucination / Fluency, 0-10)
  - LLM-as-Judge: mean over samples with overall >= 0, overall and per-category
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Sequence

import librosa
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset
from tqdm import tqdm

from src import Define
from .utils import LLMJudgeWrapper, LOCAL_JUDGE_MODEL

# Official evaluate.py CATEGORY_INSTRUCTIONS
CATEGORY_INSTRUCTIONS = {
    "sound": [
        "Describe what you hear in this audio.",
        "What sounds can you identify in this audio clip?",
    ],
    "music": [
        "Describe this music clip in detail, including genre, instrumentation, tempo, and mood.",
        "Characterize this musical excerpt with rich detail; cover genre, instrumentation, and overall atmosphere.",
    ],
    "speech": [
        "Describe the speaker and what they are saying, including their tone, emotion, and speaking style.",
        "Describe this speech audio, including the speaker's characteristics and what is being said.",
    ],
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MD_LINK_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)|\[[^\]]*\]\([^)]+\)")

CATEGORY_ORDER = ["sound", "music", "speech"]
LLM_METRIC_KEYS = (
    "llm_accuracy",
    "llm_completeness",
    "llm_hallucination",
    "llm_fluency",
    "llm_overall",
)
LLM_DISPLAY = (
    ("llm_accuracy", "accuracy"),
    ("llm_completeness", "completeness"),
    ("llm_hallucination", "hallucination"),
    ("llm_fluency", "fluency"),
    ("llm_overall", "overall"),
)


def sanitize_output(text: str) -> str:
    """Official evaluate.py sanitize_output."""
    if not text:
        return text
    cleaned = MD_LINK_PATTERN.sub("", text)
    cleaned = URL_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def official_llm_judge_prompt(
    prediction: str,
    references: list[str],
    category: str,
    transcript: str = "",
) -> str:
    """Official metrics.py evaluate_with_llm_judge prompt."""
    refs_text = "\n".join(f" - {r}" for r in references)
    category_guidance = {
        "music": (
            "For music, evaluate whether the prediction correctly identifies: "
            "genre, instrumentation, tempo, mood/atmosphere, and any vocal characteristics."
        ),
        "speech": (
            "For speech, evaluate whether the prediction correctly describes: "
            "speaker characteristics (gender, age, accent), emotional tone, "
            "speaking style, and the content of what is being said. "
            "The reference may include both a transcript and an emotional description - "
            "evaluate how well the prediction captures both aspects."
        ),
    }.get(category, (
        "For general/environmental sound, evaluate whether the prediction "
        "correctly identifies: sound sources, events, acoustic environment, "
        "and temporal patterns."
    ))
    if category == "speech" and transcript:
        category_guidance += f'\nActual transcript of the speech: "{transcript}"'

    return f"""You are an expert evaluator for audio captioning systems.

Given the ground-truth reference captions and a model's predicted caption for an audio clip,
score the prediction on the following criteria (each on a scale of 0 to 10):

1. **Accuracy** (0-10): Does the prediction correctly describe the same audio content as the references?
 Are the key sound sources, events, or attributes correct? Note: the prediction may use different
 wording than the references - focus on whether the semantic content is correct, not exact word matches.
2. **Completeness** (0-10): Does the prediction cover the main elements mentioned in the references?
 Are important details missing? A prediction that captures the most salient elements should score
 highly even if it misses minor details.
3. **Hallucination** (0-10): Does the prediction ONLY describe sounds/events that are actually
 supported by the references? 10 = no hallucination (everything described matches the references),
 0 = heavy hallucination (the prediction invents sounds, events, or attributes not present in the
 references). Penalize any fabricated content, even if the prediction also contains correct elements.
4. **Fluency** (0-10): Is the predicted caption well-written and easy to understand?
 Score the linguistic quality of the prediction itself, independent of whether it matches the references.
 10 = grammatical, concise, and easy to follow, with no redundant repetition of the same content.
 0 = ungrammatical, highly repetitive, garbled, or barely readable.

{category_guidance}

Reference captions (ground-truth):
{refs_text}

Model prediction:
 "{prediction}"

Important:
- If the prediction is empty, all scores should be 0.
- Scores must be integers from 0 to 10.
- For Accuracy, Completeness, and Hallucination, focus on semantic similarity, not surface-level word overlap.
- Fluency judges writing quality only (grammar, repetition, readability), not factual match to the references.

Respond with ONLY a JSON object, no other text:
{{"accuracy":, "completeness":, "hallucination":, "fluency":, "reasoning": "<1-2 sentence explanation>"}}"""


def parse_llm_judge_json(raw: str) -> dict[str, Any]:
    json_str = (raw or "").strip()
    if "```" in json_str:
        match = re.search(r"```(?:json)?\s*(.*?)```", json_str, re.DOTALL)
        json_str = match.group(1).strip() if match else json_str
    try:
        scores = json.loads(json_str)
    except json.JSONDecodeError:
        start, end = json_str.find("{"), json_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                scores = json.loads(json_str[start:end + 1])
            except json.JSONDecodeError:
                raise
        else:
            raise
    accuracy = max(0, min(10, float(scores.get("accuracy", 0))))
    completeness = max(0, min(10, float(scores.get("completeness", 0))))
    hallucination = max(0, min(10, float(scores.get("hallucination", 0))))
    fluency = max(0, min(10, float(scores.get("fluency", 0))))
    overall = (accuracy + completeness + hallucination + fluency) / 4.0
    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "hallucination": hallucination,
        "fluency": fluency,
        "overall": round(overall, 2),
        "reasoning": scores.get("reasoning", ""),
    }


def evaluate_with_llm_judge(
    prediction: str,
    references: list[str],
    category: str,
    llm: LLMJudgeWrapper,
    transcript: str = "",
) -> dict[str, Any]:
    """Official metrics.py evaluate_with_llm_judge via LLMJudgeWrapper."""
    prompt = official_llm_judge_prompt(prediction, references, category, transcript)
    try:
        raw = llm.generate(prompt, system_prompt="", max_new_tokens=400)
        return parse_llm_judge_json(raw)
    except Exception as e:
        print(f" LLM judge error: {e}")
        return {
            "accuracy": -1,
            "completeness": -1,
            "hallucination": -1,
            "fluency": -1,
            "overall": -1,
            "reasoning": f"Error: {e}",
        }


def _mean_nonneg(values: Sequence[Any]) -> float | None:
    """Official evaluate.py: average scores with value >= 0."""
    vals: list[float] = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x >= 0:
            vals.append(x)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _average_llm_metrics(records: list[dict]) -> dict[str, float]:
    """Official evaluate.py LLM aggregation: mean over samples with overall >= 0."""
    valid = [r for r in records if r.get("llm_overall", -1) >= 0]
    out: dict[str, float] = {}
    for key in LLM_METRIC_KEYS:
        mean = _mean_nonneg(r[key] for r in valid if key in r)
        if mean is not None:
            out[key] = mean
    return out


def _format_llm_block(title: str, scores: dict) -> list[str]:
    lines = [title]
    for key, label in LLM_DISPLAY:
        if key in scores:
            lines.append(f"  {label:20s}: {scores[key]:.4f}")
    return lines


def format_audiocapbench_results(metrics: dict) -> str:
    lines = [
        "AudioCapBench Evaluation Summary",
        "=" * 72,
    ]
    n = metrics.get("n", 0)
    corpus = metrics.get("corpus_scores") or {}
    if corpus:
        lines.extend(_format_llm_block(f"Overall ({n} samples):", corpus))
    for cat in CATEGORY_ORDER:
        cat_scores = (metrics.get("per_category") or {}).get(cat)
        if not cat_scores:
            continue
        n_cat = cat_scores.get("n", 0)
        lines.append("")
        lines.extend(_format_llm_block(f"{cat.capitalize()} ({n_cat} samples):", cat_scores))
    score = metrics.get("score")
    if score is not None:
        lines.append("")
        lines.append(f"Score: {score * 100:.2f}%")
    return "\n".join(lines) + "\n"


class AudioCapBench(object):
    def __init__(self):
        self.cache_dir = f"{Define.CACHE_DIR}/AudioCapBench"
        self.data_info_path = f"{self.cache_dir}/data_info.json"
        if not os.path.isfile(self.data_info_path):
            self.parse()
        with open(self.data_info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

    def parse(self):
        root = Define.AUDIOCAPBENCH_DIR
        src_json = os.path.join(root, "metadata.json") if root else ""
        if not os.path.isfile(src_json):
            raise FileNotFoundError(
                "AudioCapBench metadata not found. Set AUDIOCAPBENCH_DIR in .env "
                "and run `uv run python audiocapbench_download.py`."
            )
        with open(src_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        samples = metadata.get("samples") or []
        if not samples:
            raise ValueError(f"No samples in {src_json}")

        os.makedirs(f"{self.cache_dir}/wav", exist_ok=True)
        res = []
        for instance in tqdm(samples, desc="AudioCapBench"):
            src = os.path.join(root, instance["audio_file"])
            if not os.path.isfile(src):
                raise FileNotFoundError(f"Missing AudioCapBench audio: {src}")
            dest = f"{self.cache_dir}/wav/{instance['id']}.wav"
            if not os.path.isfile(dest):
                wav, _ = librosa.load(src, sr=16000)
                wavfile.write(dest, 16000, (wav * 32767).astype(np.int16))
            rec = dict(instance)
            rec["audio_path"] = dest
            res.append(rec)
        with open(self.data_info_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.info)

    def get(self, idx) -> dict:
        instance = self.info[idx]
        audio_input, _ = librosa.load(instance["audio_path"], sr=16000)
        return {
            **instance,
            "audio_input": audio_input,
        }


class AudioCapBenchSequence(Dataset):
    """Official AudioCapBench captioning eval (1,000 samples, 3 domains)."""

    def __init__(self, category: str | None = None, judge_mode: str = "") -> None:
        if category not in (None, "sound", "music", "speech"):
            raise ValueError(f"Unknown AudioCapBench category: {category}")
        self.category = category
        self.corpus = AudioCapBench()
        self.records = []
        filtered_i = 0
        for corpus_idx, rec in enumerate(self.corpus.info):
            if category and rec.get("category") != category:
                continue
            prompts = CATEGORY_INSTRUCTIONS.get(
                rec.get("category", "sound"), CATEGORY_INSTRUCTIONS["sound"]
            )
            self.records.append({
                **rec,
                "instruction": prompts[filtered_i % len(prompts)],
                "_corpus_idx": corpus_idx,
            })
            filtered_i += 1
        counts = Counter(r.get("category", "") for r in self.records)
        parts = ", ".join(
            f"{c}={counts.get(c, 0)}" for c in CATEGORY_ORDER if counts.get(c, 0)
        )
        print(f"AudioCapBench: {len(self.records)} samples ({parts})")
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
        self._metrics: dict[str, dict] = {}
        self._summary_cache: dict | None = None
        self._summary_key: tuple | None = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        sample = self.corpus.get(rec["_corpus_idx"])
        refs = rec.get("reference_captions") or [""]
        return {
            "id": rec["id"],
            "audio_input": sample["audio_input"],
            "text_input": rec["instruction"],
            "output": refs[0] if refs else "",
            "audio_path": rec["audio_path"],
            "type": rec.get("category", ""),
            "reference_captions": refs,
            "category": rec.get("category", ""),
            "transcript": rec.get("transcript", ""),
        }

    def eval(self, pred: str, gt: str, question: str = "", sample: dict = None) -> float:
        sample = sample or {}
        prediction = sanitize_output(pred or "")
        refs = sample.get("reference_captions") or ([gt] if gt else [""])
        category = sample.get("category") or sample.get("type") or "sound"
        transcript = sample.get("transcript", "")
        metrics: dict[str, Any] = {
            "prediction": prediction,
            "references": list(refs),
            "category": category,
        }
        if self.llm is not None:
            llm_scores = evaluate_with_llm_judge(
                prediction, refs, category, self.llm, transcript
            )
            metrics["llm_accuracy"] = llm_scores["accuracy"]
            metrics["llm_completeness"] = llm_scores["completeness"]
            metrics["llm_hallucination"] = llm_scores["hallucination"]
            metrics["llm_fluency"] = llm_scores["fluency"]
            metrics["llm_overall"] = llm_scores["overall"]
            metrics["llm_reasoning"] = llm_scores.get("reasoning", "")
        sid = sample.get("id") or question
        self._metrics[sid] = metrics
        self._summary_cache = None
        self._summary_key = None
        if "llm_overall" in metrics and metrics["llm_overall"] >= 0:
            return float(metrics["llm_overall"]) / 10.0
        return 0.0

    def instance_metrics(
        self,
        ids: Sequence[str],
        scores: Sequence[float],
        types: Sequence[str] | None = None,
    ) -> dict:
        key = (tuple(ids), tuple(types or []))
        if self._summary_cache is not None and self._summary_key == key:
            return self._summary_cache

        records = []
        for i, sid in enumerate(ids):
            typ = (types or [""] * len(ids))[i]
            m = dict(self._metrics.get(sid) or {})
            if "llm_overall" not in m and i < len(scores):
                m["llm_overall"] = float(scores[i]) * 10.0
            if not m.get("category"):
                m["category"] = typ
            records.append(m)

        corpus = _average_llm_metrics(records)
        per_category = {}
        for cat in CATEGORY_ORDER:
            cat_records = [m for m in records if (m.get("category") or "") == cat]
            if not cat_records:
                continue
            cat_metrics = _average_llm_metrics(cat_records)
            cat_metrics["n"] = len(cat_records)
            per_category[cat] = cat_metrics

        score = corpus.get("llm_overall", 0.0) / 10.0
        summary = {
            "score": score,
            "n": len(records),
            "corpus_scores": corpus,
            "per_category": per_category,
        }
        self._summary_key = key
        self._summary_cache = summary
        return summary

    def format_results(
        self,
        ids: Sequence[str],
        scores: Sequence[float],
        types: Sequence[str] | None = None,
    ) -> str:
        return format_audiocapbench_results(self.instance_metrics(ids, scores, types))
