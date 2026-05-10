#!/usr/bin/env python3
"""
Automated wandb experiments: compare baseline vs RAG (error-reason based historical cases).

Usage:
  python test_with_wandb.py [options]
Options:
  --config CONFIG_PATH       Path to YAML config file (default: config/config.yaml)
  --test-file TEST_FILE      Path to test JSONL file (overrides config)
  --max-samples MAX_SAMPLES Limit number of test samples to process
  --device DEVICE            Device for embedding model (default: cpu)
  --project PROJECT_NAME     Wandb project name (default: ljp-agent)
  --output-dir OUTPUT_DIR    Directory to save results (default: results/)
  --max-workers MAX_WORKERS  Maximum concurrent workers for processing (default: 10)
  --run-baseline             Run baseline experiment only
  --run-agent                Run RAG agent experiment only
  --run-all                  Run both baseline and RAG agent (default if none selected)
"""

import json
import os
import argparse
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import wandb
from src.agent.agent import LJPRAGAgent
from src.baseline.baseline import LJPBaseline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR

TERM_TOLERANCE_RATIO = 0.2
TERM_TOLERANCE_ABSOLUTE = 12
FINE_TOLERANCE_RATIO = 0.2
FINE_TOLERANCE_ABSOLUTE = 1000


def parse_args():
    parser = argparse.ArgumentParser(description="Run wandb experiments for baseline and RAG agent")
    parser.add_argument("--config", default=str(ROOT_DIR / "config" / "config.yaml"), help="Config file path (YAML)")
    parser.add_argument("--test-file", help="Test file path (overrides config)")
    parser.add_argument("--max-samples", type=int, help="Limit number of test samples")
    parser.add_argument("--device", default="cpu", help="Device for embedding model")
    parser.add_argument("--project", default="ljp-agent", help="Wandb project name")
    parser.add_argument("--output-dir", default="results/", help="Output directory for results")
    parser.add_argument("--max-workers", type=int, default=10, help="Maximum concurrent workers")
    parser.add_argument("--run-baseline", action="store_true", help="Run baseline experiment only")
    parser.add_argument("--run-agent", action="store_true", help="Run RAG agent experiment only")
    parser.add_argument("--run-all", action="store_true", help="Run both baseline and agent (default if none selected)")
    parser.add_argument("--top-k", type=int, help="Number of cases to retrieve (overrides config)")
    args = parser.parse_args()

    if not (args.run_baseline or args.run_agent):
        args.run_all = True
    return args


def load_test_data(test_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load test cases from JSONL file."""
    data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                if "meta" in case:
                    meta = case["meta"]
                    if "accusation" in meta and meta["accusation"]:
                        case["true_charges"] = meta["accusation"]
                    if "relevant_articles" in meta and meta["relevant_articles"]:
                        case["true_articles"] = meta["relevant_articles"]
                    if "term_of_imprisonment" in meta:
                        case["true_term"] = meta["term_of_imprisonment"]
                    if "punish_of_money" in meta:
                        case["true_fine"] = meta["punish_of_money"]
                data.append(case)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid line: {e}")
                continue

    seed = config.get("test", {}).get("seed", 42)
    random.seed(seed)
    random.shuffle(data)

    logger.info(f"Loaded {len(data)} test cases from {test_path}")
    return data


def extract_ground_truth(sample: Dict[str, Any]) -> Tuple[List[str], List[str], Dict, int]:
    true_charges = sample.get("true_charges", [])
    true_articles = sample.get("true_articles", [])
    true_term = sample.get("true_term", {"imprisonment": 0, "death_penalty": False, "life_imprisonment": False})
    true_fine = sample.get("true_fine", 0)
    return true_charges, true_articles, true_term, true_fine


def normalize_charge(charge: str) -> str:
    return charge.strip().rstrip("。！！").rstrip("罪")


def compute_accuracy(gold_list: List[str], pred_list: List[str]) -> bool:
    if not gold_list:
        return True
    gold_set = set(normalize_charge(c) for c in gold_list if c.strip())
    pred_set = set(normalize_charge(c) for c in pred_list if c.strip())
    return gold_set == pred_set


def compute_term_accuracy(true_term: Dict, pred_term: Dict) -> bool:
    """刑期准确率：死刑/无期布尔精确匹配，有期徒刑容忍度内匹配。"""
    if not isinstance(pred_term, dict):
        return False
    if true_term.get("death_penalty") != pred_term.get("death_penalty", False):
        return False
    if true_term.get("life_imprisonment") != pred_term.get("life_imprisonment", False):
        return False
    if not true_term.get("death_penalty") and not true_term.get("life_imprisonment"):
        true_months = true_term.get("imprisonment", 0)
        pred_months = pred_term.get("imprisonment", 0)
        tolerance = max(TERM_TOLERANCE_ABSOLUTE, true_months * TERM_TOLERANCE_RATIO)
        return abs(pred_months - true_months) <= tolerance
    return True


def compute_term_mae(true_term: Dict, pred_term: Dict) -> Optional[float]:
    """刑期MAE：仅对有期徒刑计算，死刑/无期返回None。"""
    if not isinstance(pred_term, dict):
        return None
    if true_term.get("death_penalty") or true_term.get("life_imprisonment"):
        return None
    true_months = true_term.get("imprisonment", 0)
    pred_months = pred_term.get("imprisonment", 0)
    return abs(pred_months - true_months)


def compute_fine_accuracy(true_fine: int, pred_fine: int) -> bool:
    """罚金准确率：容忍度内匹配。"""
    if not isinstance(pred_fine, (int, float)):
        return False
    tolerance = max(FINE_TOLERANCE_ABSOLUTE, true_fine * FINE_TOLERANCE_RATIO)
    return abs(pred_fine - true_fine) <= tolerance


def compute_fine_mae(true_fine: int, pred_fine: int) -> float:
    """罚金MAE。"""
    return abs(float(pred_fine) - float(true_fine))


def process_single_sample(
    idx: int,
    sample: Dict[str, Any],
    model,
    top_k: int = 3,
) -> Dict[str, Any]:
    fact = sample.get("fact", "")
    true_charges, true_articles, true_term, true_fine = extract_ground_truth(sample)

    try:
        result = model.predict(fact, top_k=top_k)
        pred_charges = result.get("pred_charges", [])
        pred_articles = result.get("pred_articles", [])
        pred_term = result.get("pred_term", {})
        pred_fine = result.get("pred_fine", 0)

        is_charge_correct = compute_accuracy(true_charges, pred_charges)
        is_article_correct = compute_accuracy(true_articles, pred_articles)
        is_term_correct = compute_term_accuracy(true_term, pred_term)
        term_mae = compute_term_mae(true_term, pred_term)
        is_fine_correct = compute_fine_accuracy(true_fine, pred_fine)
        fine_mae = compute_fine_mae(true_fine, pred_fine)

        return {
            "index": idx,
            "fact": fact,
            "true_charges": true_charges,
            "true_articles": true_articles,
            "true_term": true_term,
            "true_fine": true_fine,
            "pred_charges": pred_charges,
            "pred_articles": pred_articles,
            "pred_term": pred_term,
            "pred_fine": pred_fine,
            "full_prompt": result.get("prompt", ""),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "charge_correct": is_charge_correct,
            "article_correct": is_article_correct,
            "has_article": bool(true_articles),
            "term_correct": is_term_correct,
            "term_mae": term_mae,
            "fine_correct": is_fine_correct,
            "fine_mae": fine_mae,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Sample {idx} failed: {e}")
        return {
            "index": idx,
            "fact": fact,
            "true_charges": true_charges,
            "true_articles": true_articles,
            "true_term": true_term,
            "true_fine": true_fine,
            "pred_charges": [],
            "pred_articles": [],
            "pred_term": {},
            "pred_fine": 0,
            "full_prompt": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "charge_correct": False,
            "article_correct": False,
            "has_article": bool(true_articles),
            "term_correct": False,
            "term_mae": None,
            "fine_correct": False,
            "fine_mae": None,
            "error": str(e),
        }


def evaluate_model(
    model,
    test_data: List[Dict[str, Any]],
    max_workers: int,
    output_dir: str,
    experiment_name: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    total_samples = len(test_data)
    charge_correct = 0
    article_correct = 0
    joint_correct = 0
    total_article_samples = 0
    term_correct = 0
    term_mae_values = []
    fine_correct = 0
    fine_mae_values = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    predictions = []

    max_workers = min(max_workers, total_samples) if max_workers else total_samples

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_sample, idx, sample, model, top_k): idx
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
                if res["charge_correct"] and (not res["has_article"] or res["article_correct"]) and res["term_correct"]:
                    joint_correct += 1
                if res["term_correct"]:
                    term_correct += 1
                if res["term_mae"] is not None:
                    term_mae_values.append(res["term_mae"])
                if res["fine_correct"]:
                    fine_correct += 1
                fine_mae_values.append(res["fine_mae"])
                total_prompt_tokens += res["prompt_tokens"]
                total_completion_tokens += res["completion_tokens"]
                total_tokens += res["total_tokens"]
            else:
                predictions.append(res)

    predictions.sort(key=lambda x: x["index"])

    charge_accuracy = charge_correct / total_samples if total_samples > 0 else 0.0
    article_accuracy = article_correct / total_article_samples if total_article_samples > 0 else 1.0
    joint_accuracy = joint_correct / total_samples if total_samples > 0 else 0.0
    term_accuracy = term_correct / total_samples if total_samples > 0 else 0.0
    avg_term_mae = sum(term_mae_values) / len(term_mae_values) if term_mae_values else None
    fine_accuracy = fine_correct / total_samples if total_samples > 0 else 0.0
    avg_fine_mae = sum(fine_mae_values) / len(fine_mae_values) if fine_mae_values else 0.0

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved predictions to {output_path}")

    return {
        "charge_accuracy": charge_accuracy,
        "article_accuracy": article_accuracy,
        "joint_accuracy": joint_accuracy,
        "term_accuracy": term_accuracy,
        "avg_term_mae": avg_term_mae,
        "fine_accuracy": fine_accuracy,
        "avg_fine_mae": avg_fine_mae,
        "total_samples": total_samples,
        "total_article_samples": total_article_samples,
        "charge_correct": charge_correct,
        "article_correct": article_correct,
        "term_correct": term_correct,
        "fine_correct": fine_correct,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": total_tokens / total_samples if total_samples else 0,
        "output_path": output_path,
        "predictions": predictions,
    }


def run_baseline(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
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
    )

    with wandb.init(project=args.project, name="baseline-pure-llm") as run:
        log_dict = {
            "final_charge_acc": metrics["charge_accuracy"],
            "final_article_acc": metrics["article_accuracy"],
            "final_joint_acc": metrics["joint_accuracy"],
            "total_samples": metrics["total_samples"],
            "total_tokens": metrics["total_tokens"],
            "avg_tokens_per_sample": metrics["avg_tokens_per_sample"],
        }
        if metrics["avg_term_mae"] is not None:
            log_dict["final_term_acc"] = metrics["term_accuracy"]
            log_dict["final_term_mae"] = metrics["avg_term_mae"]
        if metrics["fine_accuracy"] is not None:
            log_dict["final_fine_acc"] = metrics["fine_accuracy"]
            log_dict["final_fine_mae"] = metrics["avg_fine_mae"]
        wandb.log(log_dict)

        artifact = wandb.Artifact("baseline_predictions", type="predictions")
        artifact.add_file(metrics["output_path"])
        run.log_artifact(artifact)

    logger.info(f"Baseline finished: charge_acc={metrics['charge_accuracy']:.4f}, article_acc={metrics['article_accuracy']:.4f}, term_acc={metrics['term_accuracy']:.4f}")
    return metrics


def run_rag(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Starting RAG agent experiment...")
    test_data = load_test_data(args.test_file or config["test"]["test_file"], config)
    if args.max_samples:
        test_data = test_data[:args.max_samples]

    model = LJPRAGAgent(config_path=args.config, device=args.device)
    top_k = args.top_k if args.top_k is not None else config.get("retriever", {}).get("top_k", 3)
    experiment_name = f"rag_top{top_k}"

    metrics = evaluate_model(
        model=model,
        test_data=test_data,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        top_k=top_k,
    )

    wandb_name = f"rag_top{top_k}"
    with wandb.init(project=args.project, name=wandb_name) as run:
        log_dict = {
            "final_charge_acc": metrics["charge_accuracy"],
            "final_article_acc": metrics["article_accuracy"],
            "final_joint_acc": metrics["joint_accuracy"],
            "total_samples": metrics["total_samples"],
            "total_tokens": metrics["total_tokens"],
            "avg_tokens_per_sample": metrics["avg_tokens_per_sample"],
        }
        if metrics["avg_term_mae"] is not None:
            log_dict["final_term_acc"] = metrics["term_accuracy"]
            log_dict["final_term_mae"] = metrics["avg_term_mae"]
        if metrics["fine_accuracy"] is not None:
            log_dict["final_fine_acc"] = metrics["fine_accuracy"]
            log_dict["final_fine_mae"] = metrics["avg_fine_mae"]
        wandb.log(log_dict)

        artifact_name = f"rag_predictions_top{top_k}"
        artifact = wandb.Artifact(artifact_name, type="predictions")
        artifact.add_file(metrics["output_path"])
        run.log_artifact(artifact)

    logger.info(f"RAG finished: charge_acc={metrics['charge_accuracy']:.4f}, article_acc={metrics['article_accuracy']:.4f}, term_acc={metrics['term_accuracy']:.4f}")
    return metrics


def main():
    args = parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.test_file:
        if "test" not in config:
            config["test"] = {}
        config["test"]["test_file"] = args.test_file

    if args.run_all or args.run_baseline:
        run_baseline(args, config)
    if args.run_all or args.run_agent:
        run_rag(args, config)

    logger.info("All experiments completed.")


if __name__ == "__main__":
    main()
