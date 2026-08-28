import os
import argparse
import time
import math
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from transformers import set_seed
import yaml
import pickle
from tqdm import tqdm
import json
import wandb

from src.systems.base import System
from src.systems.load import load_system
from src.tasks.load import get_test_task
from src.utils.tool import batchify


HEARTBEAT_INTERVAL_SEC = 60


def create_config(args) -> dict:
    """ Create a dictionary for full configuration """
    res = {"system_name": args.system_name, "task_name": None}
    if args.exp_name is not None and args.checkpoint is not None:
        res["checkpoint"] = f"{args.exp_name}/{args.checkpoint}"
        raise NotImplementedError  # currently no training yet
    res["model_config"] = {}
    for path in args.model_config:
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        res["model_config"].update(config)
    if args.start_index:
        res["start_index"] = args.start_index
    if args.max_samples is not None:
        res["max_samples"] = args.max_samples

    return res


def prepare_system(config) -> System:
    system = load_system(config["system_name"], system_config=config, checkpoint=config.get("checkpoint", None))
    system.eval()

    return system


def _flatten_numeric(obj, prefix: str = "") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            out.update(_flatten_numeric(v, key))
    elif isinstance(obj, bool):
        return out
    elif isinstance(obj, (int, float)) and math.isfinite(obj):
        out[prefix] = obj
    return out


def _init_wandb(args, config: dict):
    if args.no_wandb:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY is not set; wandb logging disabled.")
        return None

    task_label = "+".join(args.task_names)
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "sae-cd"),
        name=f"{args.system_name}-{args.output_name}-{task_label}",
        job_type="benchmark",
        config={
            "system_name": args.system_name,
            "task_names": args.task_names,
            "output_dir": args.output_dir,
            "checkpoint": args.checkpoint,
            "model_config": config.get("model_config", {}),
            "start_index": args.start_index,
            "max_samples": args.max_samples,
        },
        settings=wandb.Settings(console="off"),
        save_code=False,
    )


class Heartbeat:
    """Sparse progress pings so a stalled run is visible on wandb."""

    def __init__(self, enabled: bool, interval: float = HEARTBEAT_INTERVAL_SEC):
        self.enabled = enabled
        self.interval = interval
        self._last = 0.0
        self._t0 = time.time()
        self.total = 0
        self.task = ""

    def start_task(self, task: str, total: int):
        self.task = task
        self.total = total
        self._t0 = time.time()
        self._last = 0.0
        if self.enabled:
            wandb.summary["current_task"] = task
        self.log(0, force=True)

    def log(self, done: int, force: bool = False):
        if not self.enabled:
            return
        now = time.time()
        if not force and (now - self._last) < self.interval:
            return
        self._last = now
        wandb.log({
            "samples_done": done,
            "samples_total": self.total,
            "progress": done / self.total if self.total else 0.0,
            "elapsed_sec": now - self._t0,
        })

    def finish_task(self, tname: str, payload: dict):
        if not self.enabled:
            return
        results = payload["results"]
        metrics = {}
        scores = results.get("scores") or []
        if scores:
            metrics[f"{tname}/mean_score"] = sum(scores) / len(scores)
        summary = results.get("summary")
        if summary:
            for k, v in _flatten_numeric(summary).items():
                metrics[f"{tname}/{k}"] = v
        if metrics:
            wandb.log(metrics)
            wandb.summary.update(metrics)
        report = payload.get("report")
        if report:
            wandb.summary[f"{tname}/report"] = report.strip()


def _sample_indices(n: int, start_index: int = 0, max_samples: int | None = None) -> range:
    start = max(0, start_index)
    if start >= n:
        raise ValueError(f"--start-index {start_index} is >= dataset size {n}")
    end = n if max_samples is None else min(start + max_samples, n)
    return range(start, end)


def run_single_task(
    system: System,
    output_dir: str,
    tname: str,
    batched: bool = False,
    heartbeat: Heartbeat | None = None,
    start_index: int = 0,
    max_samples: int | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    system.eval()
    system.cuda()

    ds = get_test_task(tname)
    indices = _sample_indices(len(ds), start_index=start_index, max_samples=max_samples)
    n_run = len(indices)
    if n_run != len(ds):
        print(f"Subset: {n_run}/{len(ds)} samples (index [{indices.start}:{indices.stop}])")
    if heartbeat is not None:
        heartbeat.start_task(tname, n_run)

    basenames = []
    predictions = []
    golds = []
    scores = []
    types = []
    res_list = []
    if not batched:
        for i, idx in enumerate(tqdm(indices, total=n_run)):
            sample = ds[idx]
            res = system.inference([sample["audio_input"]], [sample["text_input"]], [sample["audio_path"]])
            res_list.append(res)

            prediction = res["prediction"]
            predictions.append(prediction)
            basenames.append(sample["id"])
            golds.append(sample["output"])
            types.append(sample.get("type", ""))
            eval_kwargs = {}
            if "sample" in getattr(ds.eval, "__code__").co_varnames:
                eval_kwargs["sample"] = sample
            score = ds.eval(prediction, sample["output"], sample["text_input"], **eval_kwargs)
            scores.append(score)
            if heartbeat is not None:
                heartbeat.log(i + 1)
    else:  # vllm, currently not supported
        bs = 64
        subset = [ds[idx] for idx in indices]
        done = 0
        for batch in tqdm(batchify(subset, batch_size=bs),
            total=(n_run + bs - 1) // bs
        ):
            audio_inputs = [sample["audio_input"] for sample in batch]
            text_inputs = [sample["text_input"] for sample in batch]
            res = system.batch_inference(audio_inputs, text_inputs)
            res_list.extend(res)

            for x, sample in zip(res, batch):
                predictions.append(x["prediction"])
                basenames.append(sample["id"])
                golds.append(sample["output"])
                types.append(sample.get("type", ""))
                score = ds.eval(x["prediction"], sample["output"])
                scores.append(score)
            done += len(batch)
            if heartbeat is not None:
                heartbeat.log(done)

    if heartbeat is not None:
        heartbeat.log(n_run, force=True)

    # Persist inference immediately so a later metric crash cannot drop predictions.
    task_dir = f"{output_dir}/{tname}"
    payload = {
        "model": system.__class__.__name__,
        "task_name": tname,
        "results": {
            "scores": scores,
            "basenames": basenames,
            "predictions": predictions,
            "golds": golds,
            "types": types,
            "all": res_list,
        }
    }
    if hasattr(ds, "_metrics"):
        llm_records = []
        for bid, typ, sc in zip(basenames, types, scores):
            src = ds._metrics.get(bid) or {}
            rec = {"category": src.get("category", typ)}
            for key in (
                "llm_accuracy",
                "llm_completeness",
                "llm_hallucination",
                "llm_fluency",
                "llm_overall",
            ):
                if key in src:
                    rec[key] = src[key]
            if "llm_overall" not in rec:
                rec["llm_overall"] = float(sc) * 10.0
            llm_records.append(rec)
        payload["results"]["llm_records"] = llm_records
    log_results(payload, task_dir)
    print(f"Saved {len(predictions)} predictions to {task_dir}")

    try:
        if hasattr(ds, "instance_metrics"):
            payload["results"]["summary"] = ds.instance_metrics(basenames, scores, types)
        if hasattr(ds, "format_results"):
            payload["report"] = ds.format_results(basenames, scores, types)
        log_results(payload, task_dir)
    except Exception:
        traceback.print_exc()
        print(
            "WARNING: metric aggregation failed after inference. "
            f"Predictions are already saved under {task_dir}"
        )

    if heartbeat is not None:
        heartbeat.finish_task(tname, payload)


def log_results(payload, output_dir: str):
    log_dir = f"{output_dir}/log"
    result_dir = f"{output_dir}/result"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # log
    with open(f'{result_dir}/results.pkl', "wb") as f:
        pickle.dump(payload, f)

    results = payload["results"]
    report = payload.get("report")
    if not report:
        score = sum(results["scores"]) / len(results["scores"])
        report = f"Score: {score * 100:.2f}%\n"
        payload["report"] = report
    print(report, end="" if report.endswith("\n") else "\n")

    with open(f'{output_dir}/results.txt', "w") as f:
        f.write(report if report.endswith("\n") else report + "\n")
    with open(f'{log_dir}/predictions.txt', "w") as f:
        for orig, pred, score, basename in zip(results["golds"], results["predictions"], results["scores"], results["basenames"]):
            f.write(f"{score:.2f}|{basename}|{orig}|{pred}\n")


def main(args):
    args.output_name = args.output_dir
    args.output_dir = f"results/{args.system_name}/{args.output_dir}"
    config = create_config(args)
    run = _init_wandb(args, config)
    heartbeat = Heartbeat(enabled=run is not None)
    try:
        system = prepare_system(config)
        print("========================== Start! ==========================")
        print(f"Output Dir: {args.output_dir}")
        print("System name: ", args.system_name)
        print("Task name: ", args.task_names)
        print("Checkpoint Path: ", args.checkpoint)
        if args.start_index or args.max_samples is not None:
            print(f"Subset: start_index={args.start_index}, max_samples={args.max_samples}")

        for tname in args.task_names:
            os.makedirs(f'{args.output_dir}/{tname}', exist_ok=True)
            config["task_name"] = tname
            with open(f'{args.output_dir}/{tname}/config.yaml', "w", encoding="utf-8") as f:
                yaml.dump(config, f, sort_keys=False)
            run_single_task(
                system,
                args.output_dir,
                tname=tname,
                heartbeat=heartbeat,
                start_index=args.start_index,
                max_samples=args.max_samples,
            )
    except Exception:
        if run is not None:
            wandb.finish(exit_code=1)
        raise
    else:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR")
    parser.add_argument('-o', '--output_dir', type=str, help="path for evaluated results")
    parser.add_argument('-s', '--system_name', type=str, help="system identifier")
    parser.add_argument('-t', '--task_names', nargs='+', help="list of task names to be evaluate")
    parser.add_argument('-n', '--exp_name', type=str, default=None)
    parser.add_argument('-c', '--checkpoint', type=str, default=None)
    parser.add_argument('--model_config', nargs='+', default=[])
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--no-wandb', action="store_true", default=False, help="disable wandb logging")
    parser.add_argument('--max-samples', type=int, default=None, help="evaluate only N samples (after --start-index)")
    parser.add_argument('--start-index', type=int, default=0, help="skip the first N samples")

    set_seed(666)
    args = parser.parse_args()
    main(args)
