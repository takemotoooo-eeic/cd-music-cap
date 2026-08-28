import os
from collections import defaultdict
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset
import json
import librosa
from tqdm import tqdm
from scipy.io import wavfile

from src import Define
from .utils import llm_as_judge, LLMJudgeWrapper, LOCAL_JUDGE_MODEL

# Official MMAU domains (evaluation.py task field). Display order: Speech Sound Music Avg.
MMAU_TASK_ORDER = [
    ("speech", "Speech"),
    ("sound", "Sound"),
    ("music", "Music"),
]


def compute_task_metrics(
    ids: Sequence[str],
    scores: Sequence[float],
    types: Sequence[str],
) -> dict:
    """Per-domain accuracy and overall Avg (micro-average, official MMAU overall)."""
    if not (len(ids) == len(scores) == len(types)):
        raise ValueError(
            f"ids/scores/types length mismatch: {len(ids)}, {len(scores)}, {len(types)}"
        )

    groups: dict[str, list[bool]] = defaultdict(list)
    for score, task in zip(scores, types):
        key = (task or "").strip().lower()
        groups[key].append(float(score) >= 0.5)

    display = dict(MMAU_TASK_ORDER)
    per_task = {}
    ordered_keys = [k for k, _ in MMAU_TASK_ORDER if k in groups]
    extra_keys = sorted(k for k in groups if k not in display and k)
    for task_key in ordered_keys + extra_keys:
        vals = groups[task_key]
        n = len(vals)
        n_correct = sum(1 for v in vals if v)
        per_task[display.get(task_key, task_key)] = {
            "key": task_key,
            "acc": n_correct / n if n else 0.0,
            "n": n,
            "n_correct": n_correct,
        }

    n_all = len(scores)
    n_correct_all = sum(1 for s in scores if float(s) >= 0.5)
    avg = n_correct_all / n_all if n_all else 0.0
    return {"score": avg, "per_task": per_task}


def format_task_results(metrics: dict) -> str:
    lines = []
    for name, stat in metrics["per_task"].items():
        lines.append(f"{name}: {stat['acc'] * 100:.2f}% (n={stat['n']})")
    lines.append(f"Avg: {metrics['score'] * 100:.2f}%")
    return "\n".join(lines) + "\n"


class MMAU_MINI(object):

    def __init__(self):
        self.cache_dir = f"{Define.CACHE_DIR}/MMAU-MINI"
        self.data_info_path = f"{self.cache_dir}/data_info.json"
        if not os.path.isfile(self.data_info_path):
            self.parse()
        with open(self.data_info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

    def parse(self):
        root = Define.MMAU_MINI
        src_json = os.path.join(root, "mmau-test-mini.json")
        audio_dir = os.path.join(root, "test-mini-audios")
        if not os.path.isfile(src_json):
            raise FileNotFoundError(
                f"MMAU mini metadata not found: {src_json}. "
                "Set MMAU_MINI in .env to the directory containing mmau-test-mini.json and test-mini-audios/."
            )
        with open(src_json, "r", encoding="utf-8") as f:
            info = json.load(f)
        os.makedirs(f"{self.cache_dir}/wav", exist_ok=True)
        res = []
        for instance in tqdm(info):
            src = os.path.join(audio_dir, f"{instance['id']}.wav")
            dest = f"{self.cache_dir}/wav/{instance['id']}.wav"
            if not os.path.isfile(src):
                raise FileNotFoundError(f"Missing MMAU audio: {src}")
            if not os.path.isfile(dest):
                wav, _ = librosa.load(src, sr=16000)
                wavfile.write(dest, 16000, (wav * 32767).astype(np.int16))
            res.append(instance)
        with open(self.data_info_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=4)

    def __len__(self):
        return len(self.info)
    
    def get(self, idx) -> dict:
        instance = self.info[idx]
        audio_input, _ = librosa.load(f"{self.cache_dir}/wav/{instance['id']}.wav", sr=16000)

        return {
            **instance,
            "audio_input": audio_input,
        }


class MMAUMINIMCQASequence(Dataset):
    def __init__(self, judge_mode: str="") -> None:
        self.corpus = MMAU_MINI()
        self.idx_seq = list(range(len(self.corpus)))
        if judge_mode == 'api':
            judge_model_name = 'gpt-4o-2024-11-20'
        elif judge_mode == 'local':
            judge_model_name = LOCAL_JUDGE_MODEL
        # Initialize LLM here
        self.llm = LLMJudgeWrapper(
            mode=judge_mode,
            model_name=judge_model_name,
            api_key=Define.API_KEY if judge_mode == "api" else None
        )


    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sample = self.corpus.get(self.idx_seq[idx])

        # full_prompt
        full_prompt = sample['question']
        for idx, choice in enumerate(sample["choices"]):
            full_prompt += f"\n{chr(65 + idx)}. {choice}"
        output = chr(65 + sample["choices"].index(sample["answer"]))
        
        inst = {
            "id": sample['id'],
            "audio_input": sample['audio_input'],
            "text_input": full_prompt,
            "output": output.lower(),
            "audio_path": f"{self.corpus.cache_dir}/wav/{sample['id']}.wav",
            "type": sample.get("task", ""),
        }
        return inst
    
    def eval(self, pred: str, gt: str, question: str = "") -> float:
        return llm_as_judge(pred=pred, gt=gt, llm=self.llm, question=question)

    def _types_for_ids(self, ids: Sequence[str]) -> list[str]:
        id_to_task = {rec["id"]: rec["task"] for rec in self.corpus.info}
        missing = [i for i in ids if i not in id_to_task]
        if missing:
            raise KeyError(f"Unknown MMAU ids (showing up to 5): {missing[:5]}")
        return [id_to_task[i] for i in ids]

    def instance_metrics(
        self,
        ids: Sequence[str],
        scores: Sequence[float],
        types: Sequence[str] | None = None,
    ) -> dict:
        if types is None or not any((t or "").strip() for t in types):
            types = self._types_for_ids(ids)
        return compute_task_metrics(ids, scores, types)

    def format_results(
        self,
        ids: Sequence[str],
        scores: Sequence[float],
        types: Sequence[str] | None = None,
    ) -> str:
        return format_task_results(self.instance_metrics(ids, scores, types))
