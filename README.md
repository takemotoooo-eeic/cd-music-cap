# LALM Contrastive Decoding Error Profiles

### Setup
- Use Python 3.11
- `pip install -r requirements.txt`
- Install Desta if needed

AudioCapBench (1,000 captions across sound / music / speech):
1. Set `AUDIOCAPBENCH_DIR` in `.env` (e.g. `/home/sarulab/kengo_takemoto/data/AudioCapBench`)
2. Download the official eval subset from HuggingFace:

```bash
uv run python audiocapbench_download.py
uv run python audiocapbench_download.py --dry-run
```

This reproduces [AudioCapBench](https://github.com/SalesforceAIResearch/AudioCapBench) `build_dataset.py` using the curated IDs in `data/audiocapbench/eval_data_ids/`.

Captioning scores other than LLM-as-Judge are computed with [aac-metrics](https://github.com/Labbeti/aac-metrics) (`bleu_1`–`bleu_4`, `meteor`, `rouge_l`, `cider_d`, `spice`, `spider`, `fense`, `spider_fl`, `fer`, `vocab`, `bert_score`). Java is required for METEOR / SPICE; jars are downloaded into `$CACHE_DIR/aac_metrics` on the first eval.

### Usage

Check available `system_name` in `src/systems/load.py` and `task_name` in `src/tasks/load.py`.

```
python run_benchmark.py -o [output_name] -s [system_name] -t [task_name] --model_config [config_path]
```

Results will be logged under `results/[system_name]/[output_name]/[task_name]/`.

#### Available Systems

| Model | System Names |
|---|---|
| Qwen2.5-Omni | `qwen`, `qwen-aad`, `qwen-acd`, `qwen-amti`, `qwen-dola` |
| Desta2.5 | `desta`, `desta-aad`, `desta-acd`, `desta-amti`, `desta-dola` |
| Audio Flamingo 3 | `af3`, `af3-aad`, `af3-acd`, `af3-amti`, `af3-dola` |

#### Available Tasks

| Task Name Pattern | Dataset |
|---|---|
| `sakura_[subject]` | SAKURA (subjects: `animal`, `emotion`, `gender`, `language`) |
| `mmau-test-mini` | MMAU Mini |
| `mmar` | MMAR |
| `aha` | AHa-Bench (official Yes/No + ASR WER eval) |
| `ah-gen` | AudioHallucination generative (CHAIR / Cover / Hal; requires `-ja` or `-jl`) |
| `audiocapbench` | [AudioCapBench](https://github.com/SalesforceAIResearch/AudioCapBench) (aac-metrics + optional LLM-as-Judge) |

Task name suffixes:
- `-ja`: Use API-based LLM judge (GPT-4o; AudioCapBench uses official GPT-4.1)
- `-jl`: Use local LLM judge (Gemma 4 E4B)
- `-m`: Multi-hop mode (SAKURA only)
- `-sound` / `-music` / `-speech`: AudioCapBench category filter (optional; default is all 1,000 samples with Overall + per-category scores)

#### Examples

```bash
# Qwen baseline on SAKURA animal with API judge
python run_benchmark.py -o test -s qwen -t sakura_animal-ja

# Desta with AAD decoding on SAKURA animal with API judge
python run_benchmark.py -o test -s desta-aad -t sakura_animal-ja --model_config config/aad.yaml

# Audio Flamingo 3 with AMTI decoding
python run_benchmark.py -o test -s af3-amti -t sakura_animal-ja --model_config config/amti.yaml

# Qwen with DoLA on MMAU
python run_benchmark.py -o test -s qwen-dola -t mmau-test-mini-ja --model_config config/dola.yaml

# Qwen on AHa-Bench with local Yes/No judge
python run_benchmark.py -o test -s qwen -t aha-jl

python run_benchmark.py -o test -s qwen -t ah-gen-jl

# Qwen on AudioCapBench (all 1,000: sound + music + speech)
python run_benchmark.py -o test -s qwen -t audiocapbench-ja

# Audio Flamingo 3 on AudioCapBench with local judge
python run_benchmark.py -o test -s af3 -t audiocapbench-jl
```

#### Config Files

Decoding method configs are under `config/`:
- `aad.yaml` — AAD 
- `acd.yaml` — ACD
- `amti.yaml` — AMTI
- `dola.yaml` — DoLA

### Analysis

```bash
python analysis/analyze_wrong_state_w_question.py <input_result.txt> <output_states.jsonl> <task_name> <mode>
```
- `mode`: `api` (GPT-4o) or `local` (Llama-3.1-8B)
