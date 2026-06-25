"""
Downloads only what we need:
  - PTB-XL: ptbxl_database.csv, scp_statements.csv, records100/ (~1.7 GB)
"""

import time  # noqa: F401
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-downloader/1.0)"}

PTBXL_BASE  = "https://physionet.org/files/ptb-xl/1.0.3"
PTBXL_DIR   = Path("data/raw/ptbxl")
MAX_WORKERS = 8


def download_file(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, stream=True, timeout=60)
            if r.status_code != 200:
                return f"ERR  {url}  → HTTP {r.status_code}"
            expected = int(r.headers.get("Content-Length", 0))
            if dest.exists() and expected > 0 and dest.stat().st_size == expected:
                return f"skip {dest.name}"
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            tmp.replace(dest)
            return f"ok   {dest.name}"
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERR  {url}  → connection failed after 3 attempts"


def download_ptbxl():
    print("\n=== PTB-XL ===")
    PTBXL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Metadata CSVs first (small, sequential)
    for fname in ["ptbxl_database.csv", "scp_statements.csv"]:
        dest = PTBXL_DIR / fname
        print(f"  downloading {fname} ...", end=" ")
        msg = download_file(f"{PTBXL_BASE}/{fname}", dest)
        print(msg)

    # 2. Parse CSV — keep only fold 10 (test set)
    db = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv")
    fold10 = db[db["strat_fold"] == 10]
    records = fold10["filename_lr"].tolist()  # e.g. records100/00000/00001_lr
    print(f"  fold 10: {len(records)} records (out of {len(db)} total)")

    # Build list of (url, dest) pairs for .dat and .hea
    tasks = []
    for rec in records:
        for ext in (".dat", ".hea"):
            url  = f"{PTBXL_BASE}/{rec}{ext}"
            dest = PTBXL_DIR / (rec + ext)
            tasks.append((url, dest))

    print(f"  {len(tasks)} files to check/download ({len(records)} records × 2 extensions)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download_file, url, dest): dest for url, dest in tasks}
        errors = []
        for f in tqdm(as_completed(futures), total=len(futures), desc="  PTB-XL records100"):
            msg = f.result()
            if msg.startswith("ERR"):
                errors.append(msg)

    if errors:
        print(f"\n  {len(errors)} errors:")
        for e in errors[:10]:
            print(f"    {e}")
    else:
        print("  PTB-XL download complete.")


if __name__ == "__main__":
    import sys
    # pass "ptbxl" 
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("all", "ptbxl"):
        download_ptbxl()

