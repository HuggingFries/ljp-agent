#!/usr/bin/env python3
"""
Analyze per-charge improvement/degradation between baseline and RAG.

Usage:
    python scripts/analyze_charge_impact.py --baseline results/baseline_results.json --rag results/rag_negative_top3_results.json --output results/charge_impact_report.txt
"""

import argparse
import json
from collections import defaultdict


def load_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'cases' in data:
        return data['cases']
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--rag', required=True)
    parser.add_argument('--output', default='charge_impact_report.txt')
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    rag = load_results(args.rag)

    # Align by index
    if len(baseline) != len(rag):
        print(f"Warning: length mismatch {len(baseline)} vs {len(rag)}")
    min_len = min(len(baseline), len(rag))

    # For each true charge (may have multiple), categorize
    improvement_by_charge = defaultdict(int)   # baseline wrong, rag correct
    degradation_by_charge = defaultdict(int)   # baseline correct, rag wrong
    baseline_wrong_by_charge = defaultdict(int)
    rag_wrong_by_charge = defaultdict(int)

    for i in range(min_len):
        b = baseline[i]
        r = rag[i]

        true_charges = b.get('true_charges', [])
        if not true_charges:
            continue

        b_correct = b.get('charge_correct', False)
        r_correct = r.get('charge_correct', False)

        # For multi-charge cases, we use the primary (first) or all? Here we use first for simplicity
        primary_charge = true_charges[0]

        baseline_wrong_by_charge[primary_charge] += int(not b_correct)
        rag_wrong_by_charge[primary_charge] += int(not r_correct)

        if not b_correct and r_correct:
            improvement_by_charge[primary_charge] += 1
        elif b_correct and not r_correct:
            degradation_by_charge[primary_charge] += 1

    # Write report
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("Charge-level impact of RAG (vs Baseline)\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Charge':<20} {'Improve':>8} {'Degrade':>8} {'Net':>8} {'BaseWrong':>8} {'RAGWrong':>8}\n")
        f.write("-" * 70 + "\n")

        all_charges = set(improvement_by_charge.keys()) | set(degradation_by_charge.keys())
        for ch in sorted(all_charges):
            imp = improvement_by_charge.get(ch, 0)
            deg = degradation_by_charge.get(ch, 0)
            net = imp - deg
            base_w = baseline_wrong_by_charge.get(ch, 0)
            rag_w = rag_wrong_by_charge.get(ch, 0)
            f.write(f"{ch:<20} {imp:>8} {deg:>8} {net:>+8} {base_w:>8} {rag_w:>8}\n")

        f.write("\nSummary:\n")
        total_improve = sum(improvement_by_charge.values())
        total_degrade = sum(degradation_by_charge.values())
        f.write(f"Total improvements (baseline wrong -> RAG correct): {total_improve}\n")
        f.write(f"Total degradations (baseline correct -> RAG wrong): {total_degrade}\n")
        f.write(f"Net change: {total_improve - total_degrade}\n")

    print(f"Report saved to {args.output}")


if __name__ == '__main__':
    main()