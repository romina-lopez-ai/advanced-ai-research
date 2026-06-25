"""
Downloads CPSC 2018 from PhysioNet Challenge 2020.
Only the cpsc_2018 subset (6,877 ECGs, g1-g7, .mat + .hea files).
Output: data/raw/cpsc2018/g1/ ... g7/

URL structure:
  base/RECORDS         → lists group dirs like training/cpsc_2018/g1/
  base/{group}RECORDS  → lists record names like A0001, A0002, ...
  base/{group}{name}.mat / .hea
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

BASE        = "https://physionet.org/files/challenge-2020/1.0.2"
CPSC_PREFIX = "training/cpsc_2018/"         # only this subset, not _extra
CPSC_DIR    = Path("data/raw/cpsc2018")
MAX_WORKERS = 8
HEADERS     = {"User-Agent": "Mozilla/5.0 (compatible; research-downloader/1.0)"}


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


def fetch_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"  ERROR fetching {url}  → HTTP {r.status_code}", file=sys.stderr)
        sys.exit(1)
    return r.text


def download_cpsc():
    print("\n=== CPSC 2018 ===")
    CPSC_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Get top-level RECORDS, filter to cpsc_2018 groups (not _extra)
    print("  fetching group list ...", end=" ", flush=True)
    all_entries = fetch_text(f"{BASE}/RECORDS").strip().splitlines()
    group_dirs  = [e for e in all_entries
                   if e.startswith(CPSC_PREFIX) and "extra" not in e]
    print(f"{len(group_dirs)} groups: {[g.split('/')[-2] for g in group_dirs]}")

    # 2. Fetch per-group RECORDS and build download task list
    tasks = []   # (url, local_dest)
    for gdir in group_dirs:
        # gdir looks like "training/cpsc_2018/g1/"
        gname = gdir.rstrip("/").split("/")[-1]    # "g1"
        rec_url  = f"{BASE}/{gdir}RECORDS"
        records  = fetch_text(rec_url).strip().splitlines()
        for rec in records:
            for ext in (".mat", ".hea"):
                url  = f"{BASE}/{gdir}{rec}{ext}"
                dest = CPSC_DIR / gname / (rec + ext)
                tasks.append((url, dest))

    total_records = len(tasks) // 2
    print(f"  {total_records} records × 2 extensions = {len(tasks)} files to check/download")

    # 3. Download in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download_file, url, dest): dest for url, dest in tasks}
        errors  = []
        for f in tqdm(as_completed(futures), total=len(futures), desc="  CPSC 2018"):
            msg = f.result()
            if msg.startswith("ERR"):
                errors.append(msg)

    if errors:
        print(f"\n  {len(errors)} errors:")
        for e in errors[:10]:
            print(f"    {e}")
    else:
        print("  CPSC 2018 download complete.")


if __name__ == "__main__":
    download_cpsc()
