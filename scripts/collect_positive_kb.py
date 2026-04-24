#!/usr/bin/env python3
"""
Extract positive examples from CAIL2018 training set.
Each charge gets up to N samples (with full fact and true labels).

Usage:
  python scripts/collect_positive_kb.py --train data/final_all_data/first_stage/train.json --output data/positive_cases/collected_positive_cases.json --per-charge 10 --seed 42
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from typing import Dict, List, Any


def load_train_data(file_path: str) -> List[Dict[str, Any]]:
    """Load CAIL2018 JSONL format data."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skip invalid line: {e}", file=sys.stderr)
    return data


def extract_charges_and_articles(case: Dict[str, Any]) -> tuple:
    """
    Extract list of charge names and list of article numbers from a case.
    Returns (charges_list, articles_list).
    """
    # 优先从 meta 中获取
    meta = case.get('meta', {})
    charges = meta.get('accusation', [])
    articles = meta.get('relevant_articles', [])
    
    # 如果 meta 中没有，尝试顶层字段（兼容某些预处理格式）
    if not charges:
        charges = case.get('charge', [])
    if not articles:
        articles = case.get('article', [])
    
    # 如果 charges 是字符串，分割成列表
    if isinstance(charges, str):
        charges = [c.strip() for c in charges.split(',') if c.strip()]
    if isinstance(articles, str):
        articles = [a.strip() for a in articles.split(',') if a.strip()]
    
    return charges, articles


def main():
    parser = argparse.ArgumentParser(description="Extract positive examples from CAIL2018 training set")
    parser.add_argument('--train', required=True, help='Path to training JSONL file')
    parser.add_argument('--output', default='positive_examples.json', help='Output JSON file')
    parser.add_argument('--per-charge', type=int, default=10, help='Number of examples per charge (default 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载所有案例
    print(f"Loading training data from {args.train}...")
    all_cases = load_train_data(args.train)
    print(f"Loaded {len(all_cases)} cases.")

    # 按罪名分组
    charge_to_cases = defaultdict(list)
    for case in all_cases:
        fact = case.get('fact', '').strip()
        if not fact:
            continue
        charges, articles = extract_charges_and_articles(case)
        if not charges:
            continue
        # 一个案例可能对应多个罪名，我们将其复制到每个罪名下（因为正例是有多种标签的案例）
        for ch in charges:
            # 存储副本，避免引用
            charge_to_cases[ch].append({
                'fact': fact,
                'charge': charges,          # 真实的所有罪名
                'article': articles,        # 真实的所有法条
                'original_index': None,     # 暂不记录
            })
    
    print(f"Found {len(charge_to_cases)} distinct charges.")

    # 每个罪名抽样
    positive_examples = []
    for charge, cases in charge_to_cases.items():
        # 去重：基于事实去重（可选，保留第一个）
        unique = {}
        for c in cases:
            unique[c['fact']] = c
        unique_cases = list(unique.values())
        
        # 随机打乱
        random.shuffle(unique_cases)
        selected = unique_cases[:args.per_charge]
        for item in selected:
            positive_examples.append({
                'charge': charge,           # 当前罪名单个，用于分组
                'true_charges': item['charge'],
                'true_articles': item['article'],
                'fact': item['fact'],
            })
        print(f"  {charge}: collected {len(selected)} out of {len(unique_cases)} unique cases")

    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(positive_examples, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(positive_examples)} positive examples to {args.output}")
    print("Summary per charge:")
    summary = defaultdict(int)
    for ex in positive_examples:
        summary[ex['charge']] += 1
    for ch, cnt in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ch}: {cnt}")


if __name__ == '__main__':
    main()