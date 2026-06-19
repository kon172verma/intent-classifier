#!/usr/bin/env python3
"""
Prepare train / val / test JSONL splits for QLoRA fine-tuning.

Split rules are identical to finetune_LoRA so results are directly comparable.

1k mode  (first 10 files from dataset_full):
    test        : sample_0001.json                          (100 examples)
    test_anchor : sample_0001.json  (same as test in 1k)    (100 examples)
    val         : sample_0010.json                          (100 examples)
    train       : sample_0002–sample_0009.json              (800 examples)

10k mode (all 100 files from dataset_full):
    test        : sample_0001 + sample_0092–sample_0100     (1 000 examples)
    test_anchor : sample_0001.json only                     (100 examples)
    val         : sample_0082–sample_0091.json              (1 000 examples)
    train       : sample_0002–sample_0081.json              (8 000 examples)

sample_0001 is always pinned to the test set so the 100-example anchor
accuracy is directly comparable between 1k and 10k experiments.

Usage
-----
    python prepare_qlora_data.py --dataset-size 1k
    python prepare_qlora_data.py --dataset-size 10k
    python prepare_qlora_data.py --dataset-size 1k \\
        --data-dir /path/to/dataset_full --out-dir /path/to/finetune_QLoRA/data
"""

import argparse
import json
from pathlib import Path

ROOT             = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = ROOT / "dataset_full"
DEFAULT_OUT_DIR  = Path(__file__).parent.parent / "data"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_files(paths: list[Path]) -> list[dict]:
    examples: list[dict] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        examples.extend(data)
    return examples


def write_jsonl(examples: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(examples):>5} examples  →  {out_path}")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare JSONL splits for QLoRA fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-size", choices=["1k", "10k"], default="1k",
        dest="dataset_size",
        help="1k = first 10 files (1 000 examples); 10k = all 100 files (10 000 examples).",
    )
    p.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Directory containing sample_XXXX.json files.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Root output directory; splits are written to a sub-folder named after dataset_size.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    out_dir  = args.out_dir / args.dataset_size
    out_dir.mkdir(parents=True, exist_ok=True)

    def fp(n: int) -> Path:
        """Return Path for sample_XXXX.json given 1-based file index n."""
        return data_dir / f"sample_{n:04d}.json"

    print(f"\n  Dataset size : {args.dataset_size}")
    print(f"  Source dir   : {data_dir}")
    print(f"  Output dir   : {out_dir}\n")

    # sample_0001 is always the anchor test file regardless of dataset size.
    anchor = load_files([fp(1)])

    if args.dataset_size == "1k":
        train = load_files([fp(i) for i in range(2, 10)])   # 0002–0009
        val   = load_files([fp(10)])                         # 0010
        test  = anchor                                       # 0001 only

    else:  # 10k
        train     = load_files([fp(i) for i in range(2, 82)])    # 0002–0081
        val       = load_files([fp(i) for i in range(82, 92)])   # 0082–0091
        test_rest = load_files([fp(i) for i in range(92, 101)])  # 0092–0100
        test      = anchor + test_rest                           # 0001 + 0092–0100

    write_jsonl(train,  out_dir / "train.jsonl")
    write_jsonl(val,    out_dir / "val.jsonl")
    write_jsonl(test,   out_dir / "test.jsonl")
    # test_anchor is always sample_0001 only — enables apples-to-apples comparison
    # between 1k and 10k runs without re-running the full test set.
    write_jsonl(anchor, out_dir / "test_anchor.jsonl")

    print(f"\n  Split summary ({args.dataset_size}):")
    print(f"    train        : {len(train)}")
    print(f"    val          : {len(val)}")
    print(f"    test         : {len(test)}")
    print(f"    test_anchor  : {len(anchor)}  (sample_0001 only)")
    print(f"\n  Done → {out_dir}")


if __name__ == "__main__":
    main()
