#!/usr/bin/env python3
import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


MITRE_PATTERN = re.compile(r"T\d{4}(?:\.\d{3})?")


def extract_ids(value) -> List[str]:
    """Extract MITRE IDs from string/list/dict-like values."""
    if value is None:
        return []
    if isinstance(value, str):
        return sorted(set(MITRE_PATTERN.findall(value)))
    if isinstance(value, list):
        out: Set[str] = set()
        for item in value:
            out.update(MITRE_PATTERN.findall(str(item)))
        return sorted(out)
    return sorted(set(MITRE_PATTERN.findall(str(value))))


def get_gold_ids(sample: Dict) -> List[str]:
    """Read truth labels from sample['gold'], fallback to sample['output']."""
    if "gold" in sample and sample["gold"] is not None:
        return extract_ids(sample["gold"])
    if "output" in sample and sample["output"] is not None:
        return extract_ids(sample["output"])
    return []


def analyze_file(path: str) -> Dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    total = len(data)
    analyzable = 0
    exact_covered = 0
    total_gold_ids = 0
    total_covered_ids = 0
    missing_counter: Counter = Counter()
    example_miss: Tuple[int, List[str], List[str], List[str]] | None = None

    for idx, sample in enumerate(data):
        instr_ids = set(extract_ids(sample.get("instruction", "")))
        gold_ids = set(get_gold_ids(sample))

        if not gold_ids:
            continue

        analyzable += 1
        covered = gold_ids.intersection(instr_ids)
        missing = sorted(gold_ids - instr_ids)

        total_gold_ids += len(gold_ids)
        total_covered_ids += len(covered)

        if not missing:
            exact_covered += 1
        else:
            missing_counter.update(missing)
            if example_miss is None:
                example_miss = (
                    idx,
                    sorted(gold_ids),
                    sorted(instr_ids),
                    missing,
                )

    id_recall = (total_covered_ids / total_gold_ids) if total_gold_ids else 0.0
    sample_coverage = (exact_covered / analyzable) if analyzable else 0.0

    result = {
        "file": path,
        "total_samples": total,
        "analyzable_samples": analyzable,
        "samples_exactly_covered": exact_covered,
        "sample_exact_coverage_rate": sample_coverage,
        "gold_id_count": total_gold_ids,
        "covered_gold_id_count": total_covered_ids,
        "gold_id_recall_in_instruction": id_recall,
        "top_missing_ids": missing_counter.most_common(20),
    }

    if example_miss is not None:
        idx, gold_ids, instr_ids, missing = example_miss
        result["first_miss_example"] = {
            "index": idx,
            "gold_ids": gold_ids,
            "instruction_ids": instr_ids[:40],
            "missing_ids": missing,
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze whether MITRE IDs in rankgpt instruction cover gold truth labels."
    )
    parser.add_argument(
        "--pattern",
        default="datasets/TechniqueRAG-Datasets/**/*rankgpt*.json",
        help="Glob pattern for dataset files.",
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="Optional path to save full report as JSON.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern, recursive=True))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    print(f"Matched {len(files)} files")
    print("-" * 88)

    all_results = []
    total_analyzable = 0
    total_exact = 0
    total_gold = 0
    total_covered = 0

    for path in files:
        r = analyze_file(path)
        all_results.append(r)
        total_analyzable += r["analyzable_samples"]
        total_exact += r["samples_exactly_covered"]
        total_gold += r["gold_id_count"]
        total_covered += r["covered_gold_id_count"]

        print(
            f"{Path(path).name}: "
            f"samples_exact={r['samples_exactly_covered']}/{r['analyzable_samples']} "
            f"({r['sample_exact_coverage_rate']:.2%}), "
            f"id_recall={r['gold_id_recall_in_instruction']:.2%}"
        )

    overall_sample_rate = (total_exact / total_analyzable) if total_analyzable else 0.0
    overall_id_recall = (total_covered / total_gold) if total_gold else 0.0

    print("-" * 88)
    print(
        f"OVERALL: samples_exact={total_exact}/{total_analyzable} "
        f"({overall_sample_rate:.2%}), id_recall={overall_id_recall:.2%}"
    )

    report = {
        "pattern": args.pattern,
        "overall": {
            "analyzable_samples": total_analyzable,
            "samples_exactly_covered": total_exact,
            "sample_exact_coverage_rate": overall_sample_rate,
            "gold_id_count": total_gold,
            "covered_gold_id_count": total_covered,
            "gold_id_recall_in_instruction": overall_id_recall,
        },
        "files": all_results,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
