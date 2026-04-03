#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Dict, List


MITRE_PATTERN = re.compile(r"T\d{4}(?:\.\d{3})?")


def extract_mitre_ids(value) -> List[str]:
    """Extract MITRE IDs from str/list and preserve first-seen order."""
    if value is None:
        return []
    if isinstance(value, list):
        text = " ".join(str(x) for x in value)
    else:
        text = str(value)
    # dict.fromkeys keeps order while deduplicating
    return list(dict.fromkeys(MITRE_PATTERN.findall(text)))


def label_to_ids(sample: Dict) -> List[str]:
    """Prefer gold; fallback to output."""
    if "gold" in sample and sample["gold"] is not None:
        ids = extract_mitre_ids(sample["gold"])
        if ids:
            return ids
    if "output" in sample and sample["output"] is not None:
        return extract_mitre_ids(sample["output"])
    return []


def convert_json_to_rows(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict] = []
    for idx, sample in enumerate(data):
        query = sample.get("input")
        if not query:
            continue
        ids = label_to_ids(sample)
        if not ids:
            continue
        rows.append(
            {
                "query": query,
                "tech_id": str(ids),  # keep format expected by run_ranking_pipeline.py (eval)
                "source_file": path.name,
                "row_idx": idx,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert rankgpt JSON datasets to TSV files for run_ranking_pipeline.py"
    )
    parser.add_argument(
        "--input_glob",
        default="datasets/TechniqueRAG-Datasets/**/*rankgpt*.json",
        help="Glob for rankgpt JSON files",
    )
    parser.add_argument(
        "--output_dir",
        default="datasets/TechniqueRAG-Datasets/tsv_rankgpt",
        help="Directory to save TSV files",
    )
    parser.add_argument(
        "--single_output",
        default="",
        help="Optional single TSV path to merge all inputs",
    )
    parser.add_argument(
        "--minimal_columns",
        action="store_true",
        help="Only output columns required by ranking pipeline: query, tech_id",
    )
    args = parser.parse_args()

    files = sorted(Path(p) for p in glob.glob(args.input_glob, recursive=True))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    merged_rows: List[Dict] = []
    for path in files:
        rows = convert_json_to_rows(path)
        total_rows += len(rows)
        merged_rows.extend(rows)

        out_path = out_dir / f"{path.stem}.tsv"
        fieldnames = ["query", "tech_id"] if args.minimal_columns else ["query", "tech_id", "source_file", "row_idx"]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            if args.minimal_columns:
                writer.writerows({"query": r["query"], "tech_id": r["tech_id"]} for r in rows)
            else:
                writer.writerows(rows)
        print(f"{path.name} -> {out_path} ({len(rows)} rows)")

    if args.single_output:
        single_out = Path(args.single_output)
        single_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["query", "tech_id"] if args.minimal_columns else ["query", "tech_id", "source_file", "row_idx"]
        with single_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            if args.minimal_columns:
                writer.writerows({"query": r["query"], "tech_id": r["tech_id"]} for r in merged_rows)
            else:
                writer.writerows(merged_rows)
        print(f"Merged TSV -> {single_out} ({len(merged_rows)} rows)")

    print(f"Done. Processed {len(files)} files, total rows: {total_rows}")


if __name__ == "__main__":
    main()
