import os
import re
from collections import defaultdict
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset
import json
import librosa
from tqdm import tqdm
from scipy.io import wavfile

from src import Define
from .utils import LLMJudgeWrapper

# Official AHa-Bench grouping (eval_metric.py): type.split("_")[0], index[:2] as sample key.
# Display names follow the paper Table 2 column order.
AHA_TASK_ORDER = [
    ("homophone", "Homo."),
    ("polysemy", "Poly."),
    ("prosodic", "Proso."),
    ("knowledge", "Knowl."),
    ("asr", "Instr."),
    ("source", "SrcNum."),
    ("existence", "Exist."),
    ("distance", "Dist."),
    ("duration", "Dur."),
    ("temporal sequence", "Temp."),
    ("repetition", "Repet."),
    ("Authenticity", "Auth."),
    ("inferred sound", "Infa."),
    ("inferredsound", "Infs."),
    ("overreliance", "Overrel."),
]

# Official AHa-Bench Yes/No matcher (gpt_eval.py).
AHA_YN_USER_PROMPT = (
    "You are an AI assistant who will help me to match an answer with two options of a question. "
    "The options are only Yes / No. "
    "You are provided with a question and an answer, "
    "and you need to find which option (Yes / No) is most similar to the answer. "
    "If the meaning of all options are significantly different from the answer, output Unknown. "
    "Your should output a single word among the following 3 choices: Yes, No, Unknown.\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
)


def aha_normalize(s: str) -> str:
    """Official AHa-Bench label normalize (eval_metric.py)."""
    s = (s or "").strip().lower()
    if "yes" in s or "是" in s:
        return "yes"
    if "no" in s or "否" in s:
        return "no"
    return s


def aha_prediction_match(output: str) -> str:
    """Reduce judge output to the single word gpt_eval.py expects (Yes / No / Unknown)."""
    text = (output or "").strip()
    match = re.match(r"(yes|no|unknown)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return text


def aha_wer(ref: str, hyp: str) -> float:
    """Official AHa-Bench WER (eval_metric.py)."""
    import jiwer
    return jiwer.wer((ref or "").strip().lower(), (hyp or "").strip().lower())


def llm_as_judge_aha(pred: str, gt: str, llm: LLMJudgeWrapper, question: str = "") -> float:
    """Official AHa-Bench Yes/No matching (gpt_eval.py + eval_metric.py)."""
    user_prompt = AHA_YN_USER_PROMPT.format(question=question, answer=pred)
    output = llm.generate(user_prompt, system_prompt="", max_new_tokens=16)
    prediction_match = aha_prediction_match(output)
    return 1.0 if aha_normalize(prediction_match) == aha_normalize(gt) else 0.0


def aha_task_type(type_str: str) -> str:
    return (type_str or "").split("_")[0]


def aha_sample_key(index: str) -> str:
    return "_".join(str(index).split("_")[:2])


def compute_instance_metrics(
    ids: Sequence[str],
    scores: Sequence[float],
    types: Sequence[str],
) -> dict:
    """Official AHa-Bench ACC: an audio is correct iff every linked question is correct.

    Returns per-task instance accuracy and their unweighted mean (paper Table 2 Mean).
    """
    if not (len(ids) == len(scores) == len(types)):
        raise ValueError(
            f"ids/scores/types length mismatch: {len(ids)}, {len(scores)}, {len(types)}"
        )

    groups: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for sid, score, type_str in zip(ids, scores, types):
        groups[aha_task_type(type_str)][aha_sample_key(sid)].append(float(score) >= 0.5)

    display = dict(AHA_TASK_ORDER)
    per_task = {}
    ordered_keys = [k for k, _ in AHA_TASK_ORDER if k in groups]
    extra_keys = sorted(k for k in groups if k not in display)
    for task_key in ordered_keys + extra_keys:
        samples = groups[task_key]
        n = len(samples)
        n_correct = sum(1 for qs in samples.values() if all(qs))
        per_task[display.get(task_key, task_key)] = {
            "key": task_key,
            "acc": n_correct / n if n else 0.0,
            "n": n,
            "n_correct": n_correct,
        }

    mean = (
        sum(v["acc"] for v in per_task.values()) / len(per_task) if per_task else 0.0
    )
    return {"score": mean, "per_task": per_task}


def format_instance_results(metrics: dict) -> str:
    lines = [f"Score: {metrics['score'] * 100:.2f}%"]
    for name, stat in metrics["per_task"].items():
        lines.append(f"{name}: {stat['acc'] * 100:.2f}% (n={stat['n']})")
    return "\n".join(lines) + "\n"


class AHA(object):
    def __init__(self):
        self.cache_dir = f"{Define.CACHE_DIR}/AHA"
        self.data_info_path = f"{self.cache_dir}/data_info.json"
        if not os.path.isfile(self.data_info_path) or not self._cache_includes_asr():
            self.parse()
        with open(self.data_info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

    def _cache_includes_asr(self) -> bool:
        try:
            with open(self.data_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            return any("asr" in str(x.get("type", "")).lower() for x in info)
        except (OSError, json.JSONDecodeError):
            return False

    def parse(self):
        root = Define.AHA_DIR
        src_json = os.path.join(root, "data_info.json")
        if not os.path.isfile(src_json):
            raise FileNotFoundError(
                f"AHa-Bench metadata not found: {src_json}. "
                "Set AHA_DIR in .env and run `uv run python aha_download.py`."
            )
        with open(src_json, "r", encoding="utf-8") as f:
            info = json.load(f)

        os.makedirs(f"{self.cache_dir}/wav", exist_ok=True)
        src_to_dest = {}
        res = []
        for instance in tqdm(info):
            src = instance["audio_path"]
            if not os.path.isfile(src):
                raise FileNotFoundError(f"Missing AHa-Bench audio: {src}")
            if src not in src_to_dest:
                rel = os.path.relpath(src, os.path.join(root, "wav"))
                dest = f"{self.cache_dir}/wav/{rel}"
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if not os.path.isfile(dest):
                    wav, _ = librosa.load(src, sr=16000)
                    wavfile.write(dest, 16000, (wav * 32767).astype(np.int16))
                src_to_dest[src] = dest
            rec = dict(instance)
            rec["audio_path"] = src_to_dest[src]
            rec["id"] = instance["index"]
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


class AHASequence(Dataset):
    """AHa-Bench: Yes/No QA plus ASR, evaluated with the official protocol."""

    def __init__(self, judge_mode: str = "") -> None:
        self.corpus = AHA()
        self.idx_seq = list(range(len(self.corpus)))
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
                model_name="microsoft/Phi-3.5-mini-instruct",
            )

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sample = self.corpus.get(self.idx_seq[idx])
        inst = {
            "id": sample["id"],
            "audio_input": sample["audio_input"],
            "text_input": sample["question"],
            "output": sample["answer"],
            "audio_path": sample["audio_path"],
            "type": sample["type"],
            "label": sample.get("label", "en"),
        }
        return inst

    def eval(self, pred: str, gt: str, question: str = "", sample: dict = None) -> float:
        type_str = (sample or {}).get("type", "")
        is_asr = "asr" in type_str.lower()
        if is_asr:
            wer = aha_wer(gt, pred)
            return 1.0 if wer < 0.1 else 0.0
        if self.llm is not None:
            return llm_as_judge_aha(pred=pred, gt=gt, llm=self.llm, question=question)
        return 1.0 if aha_normalize(pred) == aha_normalize(gt) else 0.0

    def _types_for_ids(self, ids: Sequence[str]) -> list[str]:
        id_to_type = {rec["index"]: rec["type"] for rec in self.corpus.info}
        missing = [i for i in ids if i not in id_to_type]
        if missing:
            raise KeyError(f"Unknown AHa-Bench ids (showing up to 5): {missing[:5]}")
        return [id_to_type[i] for i in ids]

    def instance_metrics(
        self,
        ids: Sequence[str],
        scores: Sequence[float],
        types: Sequence[str] | None = None,
    ) -> dict:
        if types is None:
            types = self._types_for_ids(ids)
        return compute_instance_metrics(ids, scores, types)

    def format_results(
        self,
        ids: Sequence[str],
        scores: Sequence[float],
        types: Sequence[str] | None = None,
    ) -> str:
        return format_instance_results(self.instance_metrics(ids, scores, types))
