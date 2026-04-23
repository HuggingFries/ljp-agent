#!/usr/bin/env python3
"""
Extract error cases from baseline prediction results.
Error cases are defined as samples where either:
- charge prediction is incorrect, or
- article prediction is incorrect (for samples that have ground truth articles)

Usage:
  python extract_error_cases.py --input results/baseline_results.json --output error_cases_for_test.json
"""

import argparse
import json
import logging
from pathlib import Path
from tkinter import CURRENT
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent
ROOT_DIR = CURRENT_DIR.parent
DEFAULT_INPUT_PATH = ROOT_DIR / "results/baseline_results.json"

def load_predictions(json_path: str) -> List[Dict[str, Any]]:
    """Load predictions list from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # The file might contain either a list directly or a dict with "predictions" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    else:
        raise ValueError(f"Unexpected JSON format in {json_path}. Expected list or dict with 'predictions' key.")


def is_error(sample: Dict[str, Any]) -> tuple:
    """
    Determine if a sample is an error and what type of error.
    Returns (is_error, error_type).
    error_type: 'charge_only', 'article_only', 'both', or None.
    """
    charge_correct = sample.get("charge_correct", False)
    article_correct = sample.get("article_correct", False)
    has_article = sample.get("has_article", False)
    
    # If no article ground truth, only charge matters
    if not has_article:
        if not charge_correct:
            return True, "charge_only"
        else:
            return False, None
    
    # Has article ground truth
    if not charge_correct and not article_correct:
        return True, "both"
    elif not charge_correct:
        return True, "charge_only"
    elif not article_correct:
        return True, "article_only"
    else:
        return False, None


def extract_error_cases(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter error cases and add error_type field."""
    error_cases = []
    for sample in predictions:
        is_err, err_type = is_error(sample)
        if is_err:
            sample_copy = sample.copy()
            sample_copy["error_type"] = err_type
            error_cases.append(sample_copy)
    return error_cases


def main():
    parser = argparse.ArgumentParser(description="Extract error cases from baseline prediction results.")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_PATH, help="Path to baseline results JSON file.")
    parser.add_argument("--output", "-o", default=ROOT_DIR / "results/error_cases_for_test.json", help="Output path for error cases JSON.")
    parser.add_argument("--stats-only", action="store_true", help="Only print statistics, do not save file.")
    args = parser.parse_args()
    
    # Load predictions
    logger.info(f"Loading predictions from {args.input}")
    predictions = load_predictions(args.input)
    logger.info(f"Loaded {len(predictions)} total samples")
    
    # Extract errors
    error_cases = extract_error_cases(predictions)
    
    # Statistics
    total = len(predictions)
    error_count = len(error_cases)
    charge_only = sum(1 for e in error_cases if e["error_type"] == "charge_only")
    article_only = sum(1 for e in error_cases if e["error_type"] == "article_only")
    both = sum(1 for e in error_cases if e["error_type"] == "both")
    
    logger.info("=" * 50)
    logger.info(f"Total samples: {total}")
    logger.info(f"Error samples: {error_count} ({error_count/total*100:.2f}%)")
    logger.info(f"  - Charge only: {charge_only}")
    logger.info(f"  - Article only: {article_only}")
    logger.info(f"  - Both: {both}")
    logger.info("=" * 50)
    
    if args.stats_only:
        logger.info("Stats only, no file saved.")
        return
    
    # Save error cases
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(error_cases, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(error_cases)} error cases to {output_path}")


if __name__ == "__main__":
    main()