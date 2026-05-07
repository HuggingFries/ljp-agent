#!/usr/bin/env python3
"""
Collect error cases from vanilla LLM prediction for negative knowledge base construction.
This script only finds prediction errors and saves them; error reason extraction is moved to a separate script.

Usage:
  python scripts/collect_negative_kb.py [options]

Options:
  --config CONFIG       Config file path (default: config.json)
  --train-file FILE    Training data file path (default: from config)
  --per-type N         Number of error cases per type per charge (charge/article/term), default: 1 (3 per charge)
  --output PATH        Output knowledge base path (default: data/negative_error_cases/collected_errors.json)
  --max-workers N      Parallel API workers (default: 10)
  --seed SEED          Random seed (default: 42)
  --resume-from PATH   Resume from existing checkpoint
"""

import argparse
import json
import logging
import random
import os
import sys
from pathlib import Path
from tkinter import CURRENT
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# Ensure project root is in sys.path so that src/ can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from openai import OpenAI
from src.agent.charge_matcher import ChargeMatcher

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

ERROR_TYPES = ["charge_error", "article_error", "term_error"]


def clean_charge(charge: str) -> str:
    """Clean charge name: remove trailing '罪' character"""
    charge = charge.strip()
    if charge.endswith("罪"):
        charge = charge[:-1]
    return charge


def clean_article(article: str) -> str:
    """Clean article number: keep digits only"""
    article = article.strip()
    digits = ''.join([c for c in article if c.isdigit()])
    if digits:
        return digits
    article = article.replace("《中华人民共和国刑法》", "").replace("第", "").replace("条", "").replace("款", "").strip()
    digits = ''.join([c for c in article if c.isdigit()])
    return digits if digits else article


def is_valid_term(term: dict) -> bool:
    """Check if a term prediction is structurally valid."""
    if not isinstance(term, dict):
        return False
    imp = term.get("imprisonment", 0)
    if not isinstance(imp, (int, float)) or imp < 0 or imp > 1200:
        return False
    dp = term.get("death_penalty", False)
    li = term.get("life_imprisonment", False)
    if not isinstance(dp, bool) or not isinstance(li, bool):
        return False
    if dp and li:
        return False
    if (dp or li) and imp > 0:
        return False
    return True


def is_term_accurate(true_term: dict, pred_term: dict) -> bool:
    """Check if predicted term is within tolerance: death/life exact match, imprisonment within max(20%, 12mo)."""
    if true_term.get("death_penalty") != pred_term.get("death_penalty"):
        return False
    if true_term.get("life_imprisonment") != pred_term.get("life_imprisonment"):
        return False
    true_imp = true_term.get("imprisonment", 0)
    pred_imp = pred_term.get("imprisonment", 0)
    tolerance = max(true_imp * 0.2, 12)
    return abs(true_imp - pred_imp) <= tolerance


def bare_llm_predict(
    client: OpenAI,
    model_name: str,
    fact: str,
    article_names: List[str],
) -> Tuple[List[str], List[str], Dict, str, int, int]:
    """Direct prediction with vanilla LLM, no RAG. Predicts charges, articles, and term."""
    prompt = f"""你是一个专业的法律AI助手，擅长中国刑事案件判决预测。

## 可选法条编号（必须从这里选择，不能自己编造）
{', '.join(article_names)}

## 目标案件事实
{fact}

## 任务
请预测本案的罪名、法条和刑期，**详细写出你做出该预测的推理过程**，输出格式为JSON。

注意：
- 输出标准的中国刑法罪名名称，不要编造罪名。直接输出罪名名称，不要加"罪"字后缀（例如："故意伤害" 不是 "故意伤害罪"）
- 法条必须从可选法条编号中选择，只输出编号即可
- 如果有多个罪名或法条，输出多个
- **reasoning字段必须详细写出你为什么这么判，结合案件事实和法律规定说明推理步骤**
- **刑期预测**：如果是有期徒刑，imprisonment输出月份数（如3年输出36）；如果是无期徒刑，life_imprisonment设为true；如果是死刑，death_penalty设为true。

输出格式：
{{
  "reasoning": "你的完整推理过程，详细说明为什么选择这些罪名、法条和刑期",
  "predicted_charges": ["罪名1", "罪名2"],
  "predicted_articles": ["法条编号"],
  "predicted_term": {{"imprisonment": 36, "death_penalty": false, "life_imprisonment": false}}
}}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    content = response.choices[0].message.content
    usage = response.usage

    # Parse JSON
    try:
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        predicted_charges = result.get("predicted_charges", [])
        predicted_articles = result.get("predicted_articles", [])
        pred_reasoning = result.get("reasoning", content)
        predicted_term = result.get("predicted_term", {})
    except Exception as e:
        logger.warning(f"JSON parsing failed, using raw output as reasoning: {e}")
        predicted_charges = []
        predicted_articles = []
        pred_reasoning = content
        predicted_term = {}

    return predicted_charges, predicted_articles, predicted_term, pred_reasoning, usage.prompt_tokens, usage.completion_tokens


def load_cail2018(file_path: str) -> List[dict]:
    """Load CAIL2018 format data (JSONL)"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} cases from {file_path}")
    return data


def filter_remaining_cases(
    data: List[dict],
    charge_count: Dict[str, Dict[str, int]],
    n: int,
) -> List[dict]:
    """
    Filter dataset: keep only cases containing at least one charge with an
    unfilled error-type slot. Once a charge has all error types at target,
    all cases with this charge are removed from the pool.
    """
    filtered = []
    for item in data:
        charges = item.get('charge', [])
        if not charges and 'meta' in item:
            charges = item['meta'].get('accusation', [])
        true_charges = list(map(clean_charge, charges))
        # Keep case if any charge still needs any error type
        for c in true_charges:
            if any(charge_count.get(c, {}).get(etype, 0) < n for etype in ERROR_TYPES):
                filtered.append(item)
                break
    logger.info(f"Data filtered: {len(filtered)}/{len(data)} cases remaining to process")
    return filtered


def process_single_case(
    item: dict,
    client: OpenAI,
    model_name: str,
    charge_matcher: ChargeMatcher,
    article_names: List[str],
    charge_count: Dict[str, Dict[str, int]],
    n: int,
) -> Dict[str, Any]:
    """
    Process single case in parallel: predict and classify error type.
    Pre-checks whether all charge-error-type combos are full (skip if so).
    Validates prediction quality inline: empty/short reasoning, empty charges/articles,
    invalid term count as "failed" (not counted toward collection).
    Error classification hierarchy (first match wins):
      1. charge_error: predicted charges don't match true charges
      2. article_error: charges match but articles don't
      3. term_error: charges and articles match but term outside tolerance
    """
    # Extract ground truth
    fact = item.get('fact', '')
    charges = item.get('charge', [])
    articles = item.get('article', [])
    meta = item.get('meta', {})

    if not charges and 'meta' in item:
        charges = meta.get('accusation', [])
        articles = meta.get('relevant_articles', [])

    true_charges = list(map(clean_charge, charges))
    true_charges_set = set(true_charges)
    true_articles = list(map(clean_article, articles))
    true_articles_set = set(true_articles)

    # Extract ground truth term
    term = meta.get('term_of_imprisonment', {})
    true_term = {
        "imprisonment": term.get("imprisonment", 0),
        "death_penalty": term.get("death_penalty", False),
        "life_imprisonment": term.get("life_imprisonment", False),
    }

    if not fact or not true_charges:
        return None

    # Pre-check: skip if all charges have all error types filled
    if all(
        all(charge_count.get(c, {}).get(etype, 0) >= n for etype in ERROR_TYPES)
        for c in true_charges
    ):
        return None

    # Vanilla LLM prediction
    try:
        pred_charges_list, pred_articles_list, predicted_term, pred_reasoning, prompt_tokens, completion_tokens = bare_llm_predict(
            client,
            model_name,
            fact,
            article_names,
        )

        # ---- Inline data quality check ----
        pred_reasoning_stripped = (pred_reasoning or "").strip()
        reasoning_too_short = len(pred_reasoning_stripped) < 50
        if not pred_charges_list or not pred_articles_list or reasoning_too_short:
            return {"type": "failed", "reason": "invalid_prediction"}
        if not is_valid_term(predicted_term):
            return {"type": "failed", "reason": "invalid_term"}

        pred_charges_list = charge_matcher.map_charges(pred_charges_list)
        pred_charges = set(map(clean_charge, pred_charges_list))
        pred_articles = set(map(clean_article, pred_articles_list))

        # Hierarchical error classification
        charges_correct = pred_charges == true_charges_set
        articles_correct = pred_articles == true_articles_set
        term_correct = is_term_accurate(true_term, predicted_term)

        if charges_correct and articles_correct and term_correct:
            return {"type": "correct"}

        if not charges_correct:
            error_type = "charge_error"
        elif not articles_correct:
            error_type = "article_error"
        else:
            error_type = "term_error"

        return {
            "type": "error",
            "error_type": error_type,
            "fact": fact,
            "true_charges": list(true_charges_set),
            "predicted_charges": list(pred_charges),
            "true_articles": list(true_articles_set),
            "predicted_articles": list(pred_articles),
            "true_term": true_term,
            "predicted_term": predicted_term,
            "pred_reasoning": pred_reasoning,
            "predict_prompt_tokens": prompt_tokens,
            "predict_completion_tokens": completion_tokens,
        }

    except Exception as e:
        logger.error(f"Error processing case: {e}")
        return {
            "type": "failed",
            "error": str(e),
        }


def collect_error_cases(
    client: OpenAI,
    model_name: str,
    train_data: List[dict],
    charge_matcher: ChargeMatcher,
    article_names: List[str],
    output_file: str,
    n: int = 1,
    max_workers: int = 10,
    resume_from: str = None,
) -> None:
    """
    Main collection function: find prediction errors and save them.
    Stratified collection: per charge, collect n errors of each type
    (charge_error, article_error, term_error) = 3n total per charge.
    """
    # ========== Initialize: fresh or resume ==========
    error_cases = []
    charge_count = {}
    total_processed = 0
    total_errors = 0
    total_correct = 0

    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        with open(resume_from, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        old_charge_count = existing.get("metadata", {}).get("charge_count", {})
        if old_charge_count and isinstance(list(old_charge_count.values())[0], dict):
            charge_count = old_charge_count
            error_cases = existing.get("error_cases", [])
            total_processed = existing.get("metadata", {}).get("total_processed", 0)
            total_errors = existing.get("metadata", {}).get("total_errors", 0)
            total_correct = existing.get("metadata", {}).get("total_correct", 0)
        else:
            logger.warning("Old-format checkpoint detected. Starting fresh.")

    if not charge_count:
        logger.info("Starting fresh collection")
        for c in charge_matcher.standard_charges:
            charge_count[clean_charge(c)] = {etype: 0 for etype in ERROR_TYPES}

    total_charges = len(charge_count)
    logger.info(f"Target: {n} per error type x 3 types = {n * 3} total per charge, {total_charges} total charges")
    logger.info(f"Parallel workers: {max_workers}")

    remaining_data = filter_remaining_cases(train_data, charge_count, n)
    random.shuffle(remaining_data)

    def all_charges_done():
        return all(
            all(cnt[etype] >= n for etype in ERROR_TYPES)
            for cnt in charge_count.values()
        )

    if all_charges_done():
        logger.info("All charges already completed!")
        return

    done = False
    idx = 0
    pruned_charges = {
        c for c, cnt in charge_count.items()
        if all(cnt[etype] >= n for etype in ERROR_TYPES)
    }

    if has_tqdm:
        pbar = tqdm(total=len(remaining_data), desc="Cases processed")
    else:
        pbar = None

    def get_true_charges_list(item):
        charges = item.get('charge', [])
        if not charges and 'meta' in item:
            charges = item['meta'].get('accusation', [])
        return list(map(clean_charge, charges))

    # ========== Worker pull: keep max_workers busy, prune pool as charges fill ==========
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        while not done:
            while len(futures) < max_workers and idx < len(remaining_data):
                item = remaining_data[idx]
                idx += 1
                true_charges = get_true_charges_list(item)
                if all(
                    all(charge_count.get(c, {}).get(etype, 0) >= n for etype in ERROR_TYPES)
                    for c in true_charges
                ):
                    continue
                future = executor.submit(
                    process_single_case, item, client, model_name,
                    charge_matcher, article_names, charge_count, n,
                )
                futures[future] = item

            if not futures:
                logger.info("No more items to process (all remaining charges full)")
                break

            done_futures, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

            for future in done_futures:
                item = futures.pop(future)
                result = future.result()

                total_processed += 1
                if pbar:
                    pbar.update(1)

                if result is None or result.get("type") == "failed":
                    continue

                if result["type"] == "correct":
                    total_correct += 1
                    continue

                if result["type"] == "error":
                    total_errors += 1
                    error_type = result["error_type"]
                    need_this = any(
                        charge_count.get(c, {}).get(error_type, 0) < n
                        for c in result["true_charges"]
                    )
                    if not need_this:
                        continue

                    logger.info(f"Found valid error: type={error_type}, true_charges={sorted(result['true_charges'])}, pred_charges={sorted(result['predicted_charges'])}, true_articles={sorted(result['true_articles'])}, pred_articles={sorted(result['predicted_articles'])}")

                    error_case = {
                        "fact": result["fact"],
                        "error_type": error_type,
                        "true_charges": result["true_charges"],
                        "predicted_charges": result["predicted_charges"],
                        "true_articles": result["true_articles"],
                        "predicted_articles": result["predicted_articles"],
                        "true_term": result.get("true_term", {}),
                        "predicted_term": result.get("predicted_term", {}),
                        "pred_reasoning": result["pred_reasoning"],
                        "predict_prompt_tokens": result["predict_prompt_tokens"],
                        "predict_completion_tokens": result["predict_completion_tokens"],
                    }

                    error_cases.append(error_case)

                    for c in result["true_charges"]:
                        if c in charge_count:
                            charge_count[c][error_type] += 1

                    # Save checkpoint immediately
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "metadata": {
                                "mode": "negative_kb_bare_llm",
                                "count": len(error_cases),
                                "total_processed": total_processed,
                                "total_errors": total_errors,
                                "total_correct": total_correct,
                                "per_type_target": n,
                                "charge_count": charge_count,
                                "max_workers": max_workers,
                                "resumed_from": resume_from,
                            },
                            "count": len(error_cases),
                            "error_cases": error_cases,
                        }, f, ensure_ascii=False, indent=2)

                    # Log per-charge progress (c/a/t counts)
                    type_counts = ", ".join(
                        f"{c}: c={charge_count[c].get('charge_error', 0)}/"
                        f"a={charge_count[c].get('article_error', 0)}/"
                        f"t={charge_count[c].get('term_error', 0)}"
                        for c in result["true_charges"]
                    )
                    logger.info(f"Saved ({type_counts})")

                    # Prune: any charge just became fully done? remove its cases from pool
                    for c in result["true_charges"]:
                        if c in pruned_charges:
                            continue
                        if all(charge_count[c][etype] >= n for etype in ERROR_TYPES):
                            pruned_charges.add(c)
                            n_done = len(pruned_charges)
                            logger.info(f"Charge {c} fully collected ({n_done}/{total_charges}). Pruning pool...")
                            before = len(remaining_data)
                            if idx < len(remaining_data):
                                remaining_data = filter_remaining_cases(remaining_data[idx:], charge_count, n)
                                idx = 0
                                random.shuffle(remaining_data)
                                after = len(remaining_data)
                                logger.info(f"Pool pruned: {before} -> {after}")
                                if pbar:
                                    pbar.total = after

                    if all_charges_done():
                        logger.info("All charges completed!")
                        done = True
                        break

    if pbar:
        pbar.close()

    done_count = sum(
        1 for cnt in charge_count.values()
        if all(cnt[etype] >= n for etype in ERROR_TYPES)
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "mode": "negative_kb_bare_llm",
                "count": len(error_cases),
                "total_processed": total_processed,
                "total_errors": total_errors,
                "total_correct": total_correct,
                "per_type_target": n,
                "charge_count": charge_count,
                "max_workers": max_workers,
                "resumed_from": resume_from,
                "error_rate": total_errors / total_processed if total_processed > 0 else 0,
                "done_charges": done_count,
                "total_charges": total_charges,
            },
            "count": len(error_cases),
            "error_cases": error_cases,
        }, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info(f"Collection completed!")
    logger.info(f"  Total cases processed: {total_processed}")
    logger.info(f"  Total error cases collected: {len(error_cases)}")
    logger.info(f"  Charges fully completed: {done_count}/{total_charges} ({n} per type)")
    incomplete = [
        (c, cnt) for c, cnt in charge_count.items()
        if any(cnt[etype] < n for etype in ERROR_TYPES)
    ]
    if incomplete:
        if len(incomplete) <= 20:
            logger.info(f"  Incomplete: {incomplete}")
        else:
            logger.info(f"  Incomplete: {len(incomplete)} charges (first 10: {incomplete[:10]}...)")
    logger.info(f"  Error rate: {total_errors / total_processed * 100:.2f}%")
    logger.info(f"  Parallel workers: {max_workers}")
    logger.info(f"  Saved to: {output_file}")
    logger.info("=" * 70)


def load_api_config(config: dict):
    """Load API configuration from config"""
    api_config = config.get("api", {})
    base_url = api_config.get("base_url")
    api_key_env_var = api_config.get("api_key", "DEEPSEEK_API_KEY")
    model_name = api_config.get("model_name")
    
    api_key = os.getenv(api_key_env_var)
    
    if not all([base_url, api_key, model_name]):
        raise ValueError(
            f"Missing API configuration:\n"
            f"  - base_url: {base_url}\n"
            f"  - api_key from env '{api_key_env_var}': {api_key is not None}\n"
            f"  - model_name: {model_name}\n"
            f"Please check config/config.yaml and environment variable."
        )
    
    return base_url, api_key, model_name


def main():
    parser = argparse.ArgumentParser(description='Collect error cases for negative knowledge base')
    parser.add_argument('--config', type=str, default=str(ROOT_DIR / 'config/kb_building.yaml'), help='Config file path')
    parser.add_argument('--train-file', type=str, default=None, help='Training data file path')
    parser.add_argument('--per-type', type=int, default=None, help='Error cases per type per charge (charge/article/term)')
    parser.add_argument('--output', type=str, default=None, help='Output knowledge base path')
    parser.add_argument('--max-workers', type=int, default=None, help='Parallel workers, adjust by API rate limit')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from existing checkpoint')
    args = parser.parse_args()
    
    # Read config (YAML format)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load API config from global config
    global_config_path = ROOT_DIR / 'config/config.yaml'
    with open(global_config_path, 'r', encoding='utf-8') as f:
        global_config = yaml.safe_load(f)
    
    # Load API config
    base_url, api_key, model_name = load_api_config(global_config)
    
    # Get collection config from config, command line args override
    collect_config = config.get("collection", {})
    n = args.per_type or collect_config.get("per_type", 1)
    output = args.output or collect_config.get("output", str(ROOT_DIR / "data/negative_error_cases/collected_errors.json"))
    max_workers = args.max_workers or collect_config.get("max_workers", 10)
    seed = args.seed or collect_config.get("seed", 42)
    
    # Initialize client
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # Load charge and article lists from global config paths
    accu_path = global_config.get("data", {}).get("accu_path", "data/accu.txt")
    charge_path = ROOT_DIR / accu_path
    charge_names = []
    if charge_path.exists():
        with open(charge_path, 'r', encoding='utf-8') as f:
            charge_names = [line.strip() for line in f if line.strip()]
    
    law_path = global_config.get("data", {}).get("law_path", "data/law.txt")
    article_path = ROOT_DIR / law_path
    article_names = []
    if article_path.exists():
        with open(article_path, 'r', encoding='utf-8') as f:
            article_names = [line.strip() for line in f if line.strip()]
    
    if not charge_names:
        raise ValueError(f"Charge list not found at: {charge_path}")
    
    # Initialize charge matcher
    charge_matcher = ChargeMatcher(str(charge_path))

    # Load training data
    train_file = args.train_file or collect_config.get("train_file", str(ROOT_DIR / "data/final_all_data/first_stage/train.json"))
    
    random.seed(seed)
    train_data = load_cail2018(train_file)
    logger.info(f"Loaded {len(train_data)} training cases")
    
    # Create output directory
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # Start collection
    collect_error_cases(
        client,
        model_name,
        train_data,
        charge_matcher,
        article_names,
        output,
        n=n,
        max_workers=max_workers,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()