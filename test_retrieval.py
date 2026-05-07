#!/usr/bin/env python3
"""
Test legal element weighted retrieval (adapted for unified LJPRetriever).
Randomly samples cases from test set, runs retrieval, prints results.

Usage:
  python test_retrieval.py [options]

Options:
  --config CONFIG       Config file path (default: config/config.yaml)
  --test-file FILE      Test data file (default: from config.evaluation.test_file)
  --top-k INT           Number of cases to retrieve (default: 5)
  --num-test INT        Number of random test cases to check (default: 3)
  --seed INT            Random seed (default: 42)
  --output FILE         Output result file (default: stdout)
  --device DEVICE       Device for embedding model (default: cpu)
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--test-file", help="Test data file path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of cases to retrieve")
    parser.add_argument("--num-test", type=int, default=3, help="Number of random test cases to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", help="Output file to save result")
    parser.add_argument("--device", default="cpu", help="Device for embedding model")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def load_test_cases(test_file: str) -> List[Dict[str, Any]]:
    """Load test cases from json file or jsonl file"""
    cases = []
    with open(test_file, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[' or first_char == '{':
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "data" in data:
                    return data["data"]
                elif isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                if "meta" in case and "accusation" in case["meta"] and len(case["meta"]["accusation"]) > 0:
                    case["charge_name"] = case["meta"]["accusation"][0]
                cases.append(case)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid line: {e}")
                continue

    logger.info(f"Loaded {len(cases)} test cases from {test_file}")
    return cases


def get_true_charge(case: Dict[str, Any]) -> str:
    if "charge_name" in case:
        return case["charge_name"]
    elif "true_charge" in case:
        return case["true_charge"]
    elif "gold" in case:
        return case["gold"]
    elif "charge" in case:
        if isinstance(case["charge"], list) and len(case["charge"]) > 0:
            return case["charge"][0]
        return str(case["charge"])
    elif "meta" in case and "accusation" in case["meta"] and len(case["meta"]["accusation"]) > 0:
        return case["meta"]["accusation"][0]
    return "未知"


def print_retrieval_result(
    target_case: Dict[str, Any],
    extracted_elements: Dict[str, str],
    retrieved_results: List[Dict[str, Any]],
    max_sim: float,
):
    true_charge = get_true_charge(target_case)
    fact = target_case.get("fact", "").strip()

    print("\n" + "="*80)
    print("Target Case (from test set)")
    print("-"*80)
    print(f"True charge: {true_charge}")
    print()
    if len(fact) > 500:
        fact_print = fact[:500] + "...(truncated)"
    else:
        fact_print = fact
    print(f"Fact:\n{fact_print}")
    print("-"*80)

    print("\nExtracted Legal Elements (L1)")
    print("-"*80)
    for name, value in extracted_elements.items():
        print(f"  {name}：{value}")
    print("-"*80)

    print(f"\nTop {len(retrieved_results)} Retrieved Historical Cases")
    print(f"   Max similarity: {max_sim:.4f}")
    print("="*80)

    for idx, result in enumerate(retrieved_results):
        sim = result["similarity"]
        print(f"\n### Case #{idx+1} (similarity: {sim:.4f})")

        if "L0" in result:
            L0 = result["L0"]
            fact_text = L0.get('fact', '')
            print(f"  Fact: {fact_text[:300]}..." if len(fact_text) > 300 else f"  Fact: {fact_text}")
            print(f"  True charges: {L0.get('true_charges', 'unknown')}")
            print(f"  Predicted charges: {L0.get('predicted_charges', 'unknown')}")
            L2 = result.get("L2", {})
            if L2.get("case_summary"):
                print(f"  Summary: {L2['case_summary']}")
        print("-"*60)

    print("\n" + "="*80)
    print("Retrieval complete")
    print("="*80 + "\n")


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.test_file:
        test_file = args.test_file
    else:
        test_file = config.get("evaluation", {}).get("test_file", "data/final_all_data/first_stage/test.json")

    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        sys.exit(1)

    test_cases = load_test_cases(test_file)
    if len(test_cases) == 0:
        logger.error("No test cases loaded")
        sys.exit(1)

    random.seed(args.seed)
    if args.num_test > len(test_cases):
        args.num_test = len(test_cases)
    selected_cases = random.sample(test_cases, args.num_test)
    logger.info(f"Randomly picked {len(selected_cases)} test cases")

    from src.agent.element_extractor import LegalElementExtractor
    from src.agent.retriever import LJPRetriever

    logger.info("Initializing LegalElementExtractor...")
    element_extractor = LegalElementExtractor(config_path=args.config)

    logger.info("Initializing LJPRetriever...")
    retriever = LJPRetriever(
        config_path=args.config,
        device=args.device,
    )

    if len(retriever.cases) == 0:
        logger.error("No cases loaded, did you build the index first?")
        logger.error("Run: python build_hierarchical_index.py")
        sys.exit(1)

    logger.info(f"Loaded {len(retriever.cases)} historical cases from index")

    all_results = []
    for case_idx, target_case in enumerate(selected_cases):
        print("\n\n" + "="*100)
        print(f"Test Case #{case_idx+1} / {len(selected_cases)}")
        print("="*100)

        true_charge = get_true_charge(target_case)
        target_fact = target_case.get("fact", "")
        logger.info(f"Processing test case #{case_idx+1}, true charge: {true_charge}")

        extracted_elements = element_extractor.extract(target_fact)
        logger.info(f"Extracted {len(extracted_elements)} legal elements")

        retrieved_cases = retriever.retrieve(
            target_fact=target_fact,
            target_elements=extracted_elements,
            top_k=args.top_k,
        )

        max_sim = retrieved_cases[0]["similarity"] if retrieved_cases else 0.0
        logger.info(f"Retrieval done, max similarity: {max_sim:.4f}")

        all_results.append({
            "index": case_idx,
            "target_case": target_case,
            "extracted_elements": extracted_elements,
            "retrieved_cases": retrieved_cases,
            "max_similarity": float(max_sim),
        })

        print_retrieval_result(
            target_case=target_case,
            extracted_elements=extracted_elements,
            retrieved_results=retrieved_cases,
            max_sim=max_sim,
        )

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"\nAll {len(all_results)} test results saved to {args.output}")


if __name__ == "__main__":
    main()
