#!/usr/bin/env python3
"""
Compare error cases between baseline and RAG results.
Assumes same test set order (index alignment) and same seed.

Usage:
  python scripts/compare_errors.py --baseline results/baseline_results.json --rag results/rag_negative_top3_results.json --output-dir results/compare_analysis
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Any


def load_results(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'cases' in data:
        # Some formats use {"cases": [...]}
        return data['cases']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown format in {file_path}")


def is_error(item: Dict[str, Any]) -> tuple:
    """Return (is_error, error_type)"""
    charge_ok = item.get('charge_correct', False)
    article_ok = item.get('article_correct', False)
    has_article = item.get('has_article', False)
    if not charge_ok and not article_ok:
        return True, 'both'
    elif not charge_ok:
        return True, 'charge_only'
    elif not article_ok and has_article:
        return True, 'article_only'
    else:
        return False, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, help='Baseline results JSON')
    parser.add_argument('--rag', required=True, help='RAG results JSON')
    parser.add_argument('--output-dir', default='error_analysis', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline = load_results(args.baseline)
    rag = load_results(args.rag)

    # Sanity check: same length and index match?
    if len(baseline) != len(rag):
        print(f"Warning: length mismatch: baseline {len(baseline)} vs rag {len(rag)}")
    for i, (b, r) in enumerate(zip(baseline, rag)):
        if b.get('index', i) != r.get('index', i):
            print(f"Index mismatch at position {i}: baseline index {b.get('index')}, rag index {r.get('index')}")

    # Classify each sample
    baseline_correct = []
    baseline_error = []
    rag_correct = []
    rag_error = []

    for idx, (b, r) in enumerate(zip(baseline, rag)):
        b_err, b_type = is_error(b)
        r_err, r_type = is_error(r)

        sample_info = {
            'index': idx,
            'true_charges': b.get('true_charges', []),
            'true_articles': b.get('true_articles', []),
            'fact': b.get('fact', ''),
            'baseline': {
                'pred_charges': b.get('pred_charges', []),
                'pred_articles': b.get('pred_articles', []),
                'charge_correct': b.get('charge_correct', False),
                'article_correct': b.get('article_correct', False),
                'error_type': b_type,
                'full_prompt': b.get('full_prompt', ''),
                'pred_reasoning': b.get('pred_reasoning', '')
            },
            'rag': {
                'pred_charges': r.get('pred_charges', []),
                'pred_articles': r.get('pred_articles', []),
                'charge_correct': r.get('charge_correct', False),
                'article_correct': r.get('article_correct', False),
                'error_type': r_type,
                'full_prompt': r.get('full_prompt', ''),
                'pred_reasoning': r.get('pred_reasoning', '')
            }
        }

        if not b_err:
            baseline_correct.append(sample_info)
        else:
            baseline_error.append(sample_info)

        if not r_err:
            rag_correct.append(sample_info)
        else:
            rag_error.append(sample_info)

    # Categories for comparison
    rag_only_error = []   # baseline correct, rag error
    baseline_only_error = []        # baseline error, rag correct
    both_error = []            # both error
    both_correct = []          # both correct

    for b, r in zip(baseline, rag):
        idx = b.get('index', len(both_correct))
        b_err, b_type = is_error(b)
        r_err, r_type = is_error(r)

        common = {
            'index': idx,
            'fact': b.get('fact', ''),
            'true_charges': b.get('true_charges', []),
            'true_articles': b.get('true_articles', []),
            'baseline_pred': b.get('pred_charges', []),
            'baseline_articles': b.get('pred_articles', []),
            'rag_pred': r.get('pred_charges', []),
            'rag_articles': r.get('pred_articles', []),
            'baseline_reasoning': b.get('pred_reasoning', ''),
            'rag_reasoning': r.get('pred_reasoning', '')
        }

        if not b_err and r_err:
            rag_only_error.append(common)
        elif b_err and not r_err:
            baseline_only_error.append(common)
        elif b_err and r_err:
            both_error.append(common)
        else:
            both_correct.append(common)

    # Write report
    report_path = os.path.join(args.output_dir, 'comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Error Comparison Between Baseline and RAG ===\n\n")
        f.write(f"Total samples: {len(baseline)}\n")
        f.write(f"Baseline errors: {len(baseline_error)} ({len(baseline_error)/len(baseline)*100:.2f}%)\n")
        f.write(f"RAG errors: {len(rag_error)} ({len(rag_error)/len(rag)*100:.2f}%)\n\n")
        f.write(f"Both correct: {len(both_correct)}\n")
        f.write(f"Baseline correct, RAG error (degradation): {len(rag_only_error)}\n")
        f.write(f"Baseline error, RAG correct (improvement): {len(baseline_only_error)}\n")
        f.write(f"Both error: {len(both_error)}\n\n")

        f.write("=== Degradation Cases (Baseline correct, RAG error) ===\n")
        for case in rag_only_error[:20]:  # limit
            f.write(f"Index {case['index']}\n")
            f.write(f"True: {case['true_charges']} | Pred baseline: {case['baseline_pred']} | Pred RAG: {case['rag_pred']}\n")
            f.write(f"Baseline reasoning: {case['baseline_reasoning']}\n")
            f.write(f"RAG reasoning: {case['rag_reasoning']}\n")
            f.write("-" * 50 + "\n")

        f.write("\n=== Improvement Cases (Baseline error, RAG correct) ===\n")
        for case in baseline_only_error[:20]:
            f.write(f"Index {case['index']}\n")
            f.write(f"True: {case['true_charges']} | Pred baseline: {case['baseline_pred']} | Pred RAG: {case['rag_pred']}\n")
            f.write(f"Baseline reasoning: {case['baseline_reasoning']}\n")
            f.write(f"RAG reasoning: {case['rag_reasoning']}\n")
            f.write("-" * 50 + "\n")

        f.write("\n=== Common Errors (both wrong) ===\n")
        for case in both_error[:20]:
            f.write(f"Index {case['index']}\n")
            f.write(f"True: {case['true_charges']}\n")
            f.write(f"Baseline pred: {case['baseline_pred']}, RAG pred: {case['rag_pred']}\n")
            f.write("-" * 50 + "\n")

    # Save detailed JSON files for each category
    json_categories = {
        'rag_only_errors.json': rag_only_error,
        'baseline_only_errors.json': baseline_only_error,
        'both_errors.json': both_error,
        'all_baseline_errors.json': baseline_error,
        'all_rag_errors.json': rag_error
    }

    for fname, data in json_categories.items():
        out_path = os.path.join(args.output_dir, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Report saved to {report_path}")
    print(f"Detailed error JSON files written to {args.output_dir}")


if __name__ == '__main__':
    main()