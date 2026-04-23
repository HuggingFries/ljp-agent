#!/usr/bin/env python3
"""
Automated wandb experiments: compare baseline vs RAG negative enhancement.
Only runs two experiments:
1. baseline: pure LLM, no RAG
2. RAG: our new approach with elements-weighted negative retrieval

Usage:
  python run_wandb_experiments.py [options]
"""

import json
import os
import argparse
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from click import prompt
import wandb
from src.agent.agent import LJPRAGAgent
from src.baseline.baseline import LJPBaseline


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Run wandb experiments for baseline and RAG agent")
    parser.add_argument("--config", default=str(ROOT_DIR / "config" / "config.yaml"), help="Config file path (YAML)")
    parser.add_argument("--test-file", help="Test file path (overrides config)")
    parser.add_argument("--max-samples", type=int, help="Limit number of test samples for quick experiment")
    parser.add_argument("--device", default="cpu", help="Device for embedding model")
    parser.add_argument("--project", default="ljp-agent", help="Wandb project name")
    parser.add_argument("--output-dir", default="results/", help="Output directory for results")
    parser.add_argument("--max-workers", type=int, default=10, help="Maximum concurrent workers")
    parser.add_argument("--run-baseline", action="store_true", help="Run baseline experiment only")
    parser.add_argument("--run-agent", action="store_true", help="Run RAG agent experiment only")
    parser.add_argument("--run-all", action="store_true", help="Run both baseline and agent (default if none selected)")
    args = parser.parse_args()

    if not (args.run_baseline or args.run_agent):
        args.run_all = True
    return args

def load_test_data(test_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load test cases from JSONL file (one JSON object per line).
    """
    data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                # Optional: extract multiple charges/articles from 'meta' if present
                if "meta" in case:
                    meta = case["meta"]
                    if "accusation" in meta and meta["accusation"]:
                        case["true_charges"] = meta["accusation"]
                    if "relevant_articles" in meta and meta["relevant_articles"]:
                        case["true_articles"] = meta["relevant_articles"]
                data.append(case)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid line: {e}")
                continue

    # Shuffle
    seed = config.get("test", {}).get("seed", 42)
    random.seed(seed)
    random.shuffle(data)

    logger.info(f"Loaded {len(data)} test cases from {test_path}")
    return data

def extract_ground_truth(sample: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Extract true charge and true article from a test sample.
    Returns (true_charge, true_article). Both are comma-separated strings if multiple.
    """
    # Charge extraction
    true_charges = sample.get("true_charges", [])  # Default to empty list if not found

    # Article extraction
    true_articles = sample.get("true_articles", [])  # Default to empty string if not found
    return true_charges, true_articles


def normalize_charge(charge: str) -> str:
    """Remove trailing punctuation and '罪' for consistent comparison."""
    return charge.strip().rstrip("。！！").rstrip("罪")


def compute_accuracy(gold_list: List[str], pred_list: List[str]) -> bool:
    """
    Compute exact set accuracy between gold (comma-separated) and predicted list.
    Returns True if the sets match exactly.
    """
    if not gold_list:
        # No ground truth, treat as correct (or could be ignored)
        print("No true charges/articles")
        return True
    gold_set = set(normalize_charge(c) for c in gold_list if c.strip())
    pred_set = set(normalize_charge(c) for c in pred_list if c.strip())
    return gold_set == pred_set


def process_single_sample(
    idx: int,
    sample: Dict[str, Any],
    model,  # can be LJPBaseline or LJPRAGAgent
    use_rag: bool = False,
) -> Dict[str, Any]:
    """
    Run prediction on one sample and return metrics.
    """
    fact = sample.get("fact", "")
    true_charges, true_articles = extract_ground_truth(sample)

    try:
        result = model.predict(fact)
        pred_charges = result.get("pred_charges", [])
        pred_articles = result.get("pred_articles", [])
        pred_reasoning = result.get("pred_reasoning", "")

        is_charge_correct = compute_accuracy(true_charges, pred_charges)
        is_article_correct = compute_accuracy(true_articles, pred_articles)

        return {
            "index": idx,
            "fact": fact,
            "true_charges": true_charges,
            "true_articles": true_articles,
            "pred_charges": pred_charges,
            "pred_articles": pred_articles,
            "full_prompt": result.get("prompt", ""),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "charge_correct": is_charge_correct,
            "article_correct": is_article_correct,
            "has_article": bool(true_articles),
            "error": None,
            "pred_reasoning": pred_reasoning,
        }
    except Exception as e:
        logger.error(f"Sample {idx} failed: {e}")
        return {
            "index": idx,
            "fact": fact,
            "true_charges": true_charges,
            "true_articles": true_articles,
            "pred_charges": [],
            "pred_articles": [],
            "full_prompt": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "charge_correct": False,
            "article_correct": False,
            "has_article": bool(true_articles),
            "error": str(e),
            "pred_reasoning": "",
        }


def evaluate_model(
    model,
    test_data: List[Dict[str, Any]],
    max_workers: int,
    output_dir: str,
    experiment_name: str,
    use_rag: bool = False,
    cumulative_step: int = 10,
) -> Dict[str, Any]:
    """
    Generic test function for any model with .predict() method.
    Returns aggregated metrics and saves predictions to JSON.
    """
    total_samples = len(test_data)
    charge_correct = 0
    article_correct = 0
    total_article_samples = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    predictions = []

    max_workers = min(max_workers, total_samples) if max_workers else total_samples

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_sample, idx, sample, model, use_rag): idx
            for idx, sample in enumerate(test_data)
        }
        for future in as_completed(futures):
            res = future.result()
            if res["error"] is None:
                predictions.append(res)
                if res["charge_correct"]:
                    charge_correct += 1
                if res["has_article"]:
                    total_article_samples += 1
                    if res["article_correct"]:
                        article_correct += 1
                total_prompt_tokens += res["prompt_tokens"]
                total_completion_tokens += res["completion_tokens"]
                total_tokens += res["total_tokens"]
            else:
                # Still count as incorrect but include in predictions for debugging
                predictions.append(res)

    # Sort predictions by original index
    predictions.sort(key=lambda x: x["index"])

    charge_accuracy = charge_correct / total_samples if total_samples > 0 else 0.0
    article_accuracy = article_correct / total_article_samples if total_article_samples > 0 else 1.0
    combined_accuracy = (charge_accuracy + article_accuracy) / 2

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved predictions to {output_path}")

    return {
        "charge_accuracy": charge_accuracy,
        "article_accuracy": article_accuracy,
        "combined_accuracy": combined_accuracy,
        "total_samples": total_samples,
        "total_article_samples": total_article_samples,
        "charge_correct": charge_correct,
        "article_correct": article_correct,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": total_tokens / total_samples if total_samples else 0,
        "output_path": output_path,
        "predictions": predictions,
    }


def run_baseline(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run baseline experiment (pure LLM without RAG)."""
    logger.info("Starting baseline experiment...")
    test_data = load_test_data(args.test_file or config["test"]["test_file"], config)
    if args.max_samples:
        test_data = test_data[:args.max_samples]

    model = LJPBaseline(config_path=args.config)
    metrics = evaluate_model(
        model=model,
        test_data=test_data,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        experiment_name="baseline",
        use_rag=False,
    )

    # Wandb logging
    with wandb.init(project=args.project, name="baseline-pure-llm") as run:
        wandb.log({
            "final_charge_acc": metrics["charge_accuracy"],
            "final_article_acc": metrics["article_accuracy"],
            "final_combined_acc": metrics["combined_accuracy"],
            "total_samples": metrics["total_samples"],
            "total_prompt_tokens": metrics["total_prompt_tokens"],
            "total_completion_tokens": metrics["total_completion_tokens"],
            "total_tokens": metrics["total_tokens"],
            "avg_tokens_per_sample": metrics["avg_tokens_per_sample"],
        })
        # Save predictions as artifact
        artifact = wandb.Artifact("baseline_predictions", type="predictions")
        artifact.add_file(metrics["output_path"])
        run.log_artifact(artifact)

    logger.info(f"Baseline finished: charge_acc={metrics['charge_accuracy']:.4f}, article_acc={metrics['article_accuracy']:.4f}")
    return metrics


def run_rag(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run RAG agent experiment (with negative retrieval)."""
    logger.info("Starting RAG agent experiment...")
    test_data = load_test_data(args.test_file or config["test"]["test_file"], config)
    if args.max_samples:
        test_data = test_data[:args.max_samples]

    model = LJPRAGAgent(config_path=args.config, device=args.device)
    # Get top-k from config.
    top_k = config.get("retriever", {}).get("top_k", 3)

    metrics = evaluate_model(
        model=model,
        test_data=test_data,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        experiment_name=f"rag_top{top_k}",
        use_rag=True,
    )

    with wandb.init(project=args.project, name=f"rag-elements-top{top_k}") as run:
        wandb.log({
            "final_charge_acc": metrics["charge_accuracy"],
            "final_article_acc": metrics["article_accuracy"],
            "final_combined_acc": metrics["combined_accuracy"],
            #"top_k": args.top_k,
            "total_samples": metrics["total_samples"],
            #"total_prompt_tokens": metrics["total_prompt_tokens"],
            #"total_completion_tokens": metrics["total_completion_tokens"],
            "total_tokens": metrics["total_tokens"],
            "avg_tokens_per_sample": metrics["avg_tokens_per_sample"],
        })
        artifact = wandb.Artifact(f"rag_predictions_top{top_k}", type="predictions")
        artifact.add_file(metrics["output_path"])
        run.log_artifact(artifact)

    logger.info(f"RAG finished: charge_acc={metrics['charge_accuracy']:.4f}, article_acc={metrics['article_accuracy']:.4f}")
    return metrics


def main():
    args = parse_args()

    # Load YAML config
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Override test file if provided
    if args.test_file:
        if "test" not in config:
            config["test"] = {}
        config["test"]["test_file"] = args.test_file

    # Run selected experiments
    if args.run_all or args.run_baseline:
        run_baseline(args, config)
    if args.run_all or args.run_agent:
        run_rag(args, config)

    logger.info("All experiments completed.")


if __name__ == "__main__":
    main()