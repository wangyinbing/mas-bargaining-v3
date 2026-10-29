"""Merge the re-run free-dialogue arm (after the context bug fix) into the main output.

Usage: python v3/merge_freetext_fix.py
- backs up the current free_text rows
- replaces them with output_v3/free_fix rows
- overwrites the 300 free_text transcripts in output_v3/main/transcripts
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "output_v3" / "main"
FIX = ROOT / "output_v3" / "free_fix"

main_csv = pd.read_csv(MAIN / "episodes.csv")
fix_csv = pd.read_csv(FIX / "episodes.csv")

assert (fix_csv.protocol == "free_text").all(), "free_fix should contain only free_text"
assert len(fix_csv) == 300, f"expected 300 new episodes, got {len(fix_csv)}"

backup = ROOT / "output_v3" / "_backup_freetext_bug"
backup.mkdir(parents=True, exist_ok=True)
if not (backup / "episodes_with_bug.csv").exists():
    shutil.copy2(MAIN / "episodes.csv", backup / "episodes_with_bug.csv")

kept = main_csv[main_csv.protocol != "free_text"]
merged = pd.concat([kept, fix_csv], ignore_index=True)
merged.to_csv(MAIN / "episodes.csv", index=False)

n = 0
for f in (FIX / "transcripts").glob("*free_text*.json"):
    shutil.copy2(f, MAIN / "transcripts" / f.name)
    n += 1

print(f"kept {len(kept)} structured rows + {len(fix_csv)} new free-text rows = {len(merged)}")
print(f"copied {n} transcripts")
