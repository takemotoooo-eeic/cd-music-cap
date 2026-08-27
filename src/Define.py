import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

CACHE_DIR = os.getenv("CACHE_DIR", str(_ROOT / "_cache"))
SAKURA_DIR = os.getenv("SAKURA_DIR", "")
MMAU_MINI = os.getenv("MMAU_MINI", "")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
