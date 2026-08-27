import json
from datasets import load_dataset
import soundfile as sf
from pathlib import Path

out = Path("/home/sarulab/kengo_takemoto/data/MMAU/test-mini-audios")
out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("gamma-lab-umd/MMAU-test-mini", split="test")
for row in ds:
    attrs = row["other_attributes"]
    if isinstance(attrs, str):
        attrs = json.loads(attrs)
    uid = attrs["id"]
    audio = row["context"]
    dest = out / f"{uid}.wav"
    if dest.exists():
        continue
    sf.write(dest, audio["array"], audio["sampling_rate"])
