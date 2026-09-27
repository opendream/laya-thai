"""Merge the train-only JSONL files into data/train.jsonl and write a manifest."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "train.jsonl"
MANIFEST = ROOT / "data" / "train_manifest.json"
SOURCES = (
    "massive_th.train.jsonl",
    "xnli_th.train.jsonl",
    "wongnai.train.jsonl",
    "prachathai.train.jsonl",
    "massive_en.train.jsonl",
)


def source_paths(root: Path = ROOT) -> list[Path]:
    return [root / "data" / name for name in SOURCES]


def merge(root: Path = ROOT) -> dict:
    paths = source_paths(root)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing train files: {missing}")
    out = root / "data" / "train.jsonl"
    manifest_path = root / "data" / "train_manifest.json"
    counts = []
    total = 0
    with out.open("w", encoding="utf-8") as dest:
        for path in paths:
            n = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        dest.write(line if line.endswith("\n") else line + "\n")
                        n += 1
            counts.append({"file": path.name, "rows": n})
            total += n
    manifest = {"files": counts, "total": total}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"train_file={out} total={total}")
    for item in counts:
        print(f"  {item['file']} rows={item['rows']}")
    return manifest


if __name__ == "__main__":
    merge()
