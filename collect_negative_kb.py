#!/usr/bin/env python3
"""
Collect error cases from vanilla LLM prediction for negative knowledge base construction.

Usage:
  python collect_negative_kb.py [options]

Options:
  --config CONFIG       Config file path (default: config.json)
  --train-file FILE    Training data file path (default: from config)
  --per-charge N       Number of error cases to collect per charge (default: 3)
  --output PATH        Output knowledge base path (default: data/negative_error_cases/negative_kb_bare_llm.json)
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
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# 尝试导入tqdm进度条
try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def bare_llm_predict(
    client: OpenAI,
    model_name: str,
    fact: str,
    charge_names: List[str],
    article_names: List[str],
) -> Tuple[List[str], List[str], str, int, int]:
    """Direct prediction with vanilla LLM, no RAG"""
    prompt = f"""你是一个专业的法律AI助手，擅长中国刑事案件判决预测。

## 可选罪名列表（必须从这里选择，不能自己编造）
{', '.join(charge_names)}

## 可选法条编号（必须从这里选择，不能自己编造）
{', '.join(article_names)}

## 目标案件事实
{fact}

## 任务
请预测本案的罪名和法条，输出格式为JSON。

注意：
- 罪名必须从可选罪名列表中选择，直接输出名称，不要加"罪"字后缀（例如："故意伤害" 不是 "故意伤害罪"）
- 法条必须从可选法条编号中选择，只输出编号即可
- 如果有多个罪名或法条，输出多个

输出格式：
{{
  "reasoning": "你的推理过程",
  "predicted_charges": ["罪名1", "罪名2"],
  "predicted_articles": ["法条编号"],
  "predicted_judgment": "判决结果描述"
}}
"""
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    content = response.choices[0].message.content
    usage = response.usage
    
    # 解析JSON
    try:
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        predicted_charges = result.get("predicted_charges", [])
        predicted_articles = result.get("predicted_articles", [])
        predicted_judgment = result.get("predicted_judgment", "")
    except Exception as e:
        logger.warning(f"JSON解析失败，使用原始输出: {e}")
        predicted_charges = []
        predicted_articles = []
        predicted_judgment = content
    
    return predicted_charges, predicted_articles, predicted_judgment, usage.prompt_tokens, usage.completion_tokens


def extract_error_reason(
    client: OpenAI,
    model_name: str,
    fact: str,
    true_charges: List[str],
    pred_charges: List[str],
    pred_judgment: str,
) -> str:
    """Ask LLM to extract abstract error reason"""
    prompt = f"""请你作为法律AI专家，分析这个判决预测错误的案例，提取抽象、概括的错误原因。

## 案件事实
{fact}

## 真实正确罪名
{', '.join(true_charges)}

## AI预测错误罪名
{', '.join(pred_charges)}

## AI的推理过程
{pred_judgment}

## 任务
请你提炼出1-3句话抽象概括错误原因，重点关注：
1. 结合AI的推理过程，为什么模型会预测错误？是哪些事实特征导致混淆？
2. 这个案例容易和哪个罪名混淆？错误的关键点在哪里？

要求：
- 抽象概括，不要复述完整事实
- 只保留错误机制和混淆点的核心信息
- 语言简洁，控制在100字以内

输出格式：直接输出错误原因，不要其他内容。
"""
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    reason = response.choices[0].message.content.strip()
    return reason


def load_cail2018(file_path: str) -> List[dict]:
    """Load CAIL2018 format data"""
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
    charge_count: Dict[str, int],
    per_charge_target: int,
) -> List[dict]:
    """Filter dataset: keep only cases containing at least one unfilled charge
    Once a charge reaches target, all cases with this charge are removed
    to avoid unnecessary computation and speed up collection.
    """
    filtered = []
    for item in data:
        charges = item.get('charge', [])
        if not charges and 'meta' in item:
            charges = item['meta'].get('accusation', [])
        true_charges = list(map(clean_charge, charges))
        # 只要有一个罪名没满，就保留这个案件
        for c in true_charges:
            if charge_count.get(c, 0) < per_charge_target:
                filtered.append(item)
                break
    logger.info(f"数据集过滤完成：剩余 {len(filtered)}/{len(data)} 个案件需要处理")
    return filtered


def process_single_case(
    item: dict,
    client: OpenAI,
    model_name: str,
    charge_names: List[str],
    article_names: List[str],
) -> Dict[str, Any]:
    """Process single case in parallel: predict + check for error"""
    # 读取原始数据
    fact = item.get('fact', '')
    charges = item.get('charge', [])
    articles = item.get('article', [])
    
    if not charges and 'meta' in item:
        charges = item['meta'].get('accusation', [])
        articles = item['meta'].get('relevant_articles', [])
    
    true_charges = list(map(clean_charge, charges))
    true_charges_set = set(true_charges)
    true_articles = list(map(clean_article, articles))
    true_articles_set = set(true_articles)
    
    if not fact or not true_charges:
        return None
    
    # 裸LLM预测
    try:
        pred_charges_list, pred_articles_list, pred_judgment, prompt_tokens, completion_tokens = bare_llm_predict(
            client,
            model_name,
            fact,
            charge_names,
            article_names,
        )
        pred_charges = set(map(clean_charge, pred_charges_list))
        pred_articles = set(map(clean_article, pred_articles_list))
        
        # 判断是否预测错误
        if pred_charges == true_charges_set:
            # 预测正确，不需要
            return {"type": "correct"}
        
        # 预测错误，返回数据待收集
        return {
            "type": "error",
            "fact": fact,
            "true_charges": list(true_charges_set),
            "predicted_charges": list(pred_charges),
            "true_articles": list(true_articles_set),
            "predicted_articles": list(pred_articles),
            "predicted_judgment": pred_judgment,
            "predict_prompt_tokens": prompt_tokens,
            "predict_completion_tokens": completion_tokens,
        }
    
    except Exception as e:
        logger.error(f"处理案件出错: {e}")
        return {
            "type": "failed",
            "error": str(e),
        }


def collect_error_cases(
    client: OpenAI,
    model_name: str,
    train_data: List[dict],
    charge_names: List[str],
    article_names: List[str],
    output_file: str,
    per_charge_target: int = 3,
    max_workers: int = 10,
    resume_from: str = None,
) -> None:
    """
    Main collection function: support fresh collection and resume from checkpoint
    
    Optimization:
    - After a charge reaches target number of errors, all cases of this
      charge are filtered out from remaining processing
    - This ensures later batches only process needed cases, avoids empty run
    """
    # ========== Initialize: fresh or resume ==========
    if resume_from and os.path.exists(resume_from):
        # Resume from existing checkpoint
        logger.info(f"Resuming from checkpoint: {resume_from}")
        with open(resume_from, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        error_cases = existing.get("error_cases", [])
        charge_count = existing.get("metadata", {}).get("charge_count", {})
        total_processed = existing.get("metadata", {}).get("total_processed", 0)
        total_errors = existing.get("metadata", {}).get("total_errors", 0)
        total_correct = existing.get("metadata", {}).get("total_correct", 0)
    else:
        # Start fresh collection
        logger.info("Starting fresh collection")
        error_cases = []
        charge_count = {}
        for c in charge_names:
            charge_count[clean_charge(c)] = 0
        total_processed = 0
        total_errors = 0
        total_correct = 0
    
    total_charges = len(charge_count)
    logger.info(f"Target: {per_charge_target} error cases per charge, {total_charges} total charges")
    logger.info(f"Parallel workers: {max_workers}")
    
    # Filter out completed charges initially
    remaining_data = filter_remaining_cases(train_data, charge_count, per_charge_target)
    
    # Shuffle
    random.shuffle(remaining_data)
    
    # Check if done
    def all_charges_done():
        return all(cnt >= per_charge_target for cnt in charge_count.values())
    
    if all_charges_done():
        logger.info("🎉 All charges completed!")
        return
    
    done = False
    batch_size = max_workers * 5
    current_idx = 0
    
    if has_tqdm:
        pbar = tqdm(total=len(remaining_data), desc="Cases processed")
    else:
        pbar = None
    
    # ========== Main loop: dynamic batching ==========
    while current_idx < len(remaining_data) and not done:
        # Get current batch
        batch_end = min(current_idx + batch_size, len(remaining_data))
        batch = remaining_data[current_idx:batch_end]
        
        if not batch:
            # No more cases to process
            break
        
        logger.info(f"Processing batch: {len(batch)} cases ({current_idx}/{len(remaining_data)})")
        
        # Parallel process current batch
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
            futures = []
            for item in batch:
                future = executor.submit(
                    process_single_case,
                    item,
                    client,
                    model_name,
                    charge_names,
                    article_names,
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                total_processed += 1
                result = future.result()
                
                if result is None or result["type"] == "failed":
                    continue
                
                if result["type"] == "correct":
                    total_correct += 1
                    continue
                
                elif result["type"] == "error":
                    # Prediction error, check if we still need this charge
                    total_errors += 1
                    need_this = False
                    for c in result["true_charges"]:
                        if charge_count.get(c, 0) < per_charge_target:
                            need_this = True
                            break
                    
                    if not need_this:
                        logger.info(f"Skip: all charges in this case are already full")
                        continue
                    
                    # Found needed error -> extract error reason
                    logger.info(f"Found valid error: true={sorted(result['true_charges'])}, pred={sorted(result['predicted_charges'])}")
                    
                    error_reason = extract_error_reason(
                        client,
                        model_name,
                        result["fact"],
                        result["true_charges"],
                        result["predicted_charges"],
                        result["predicted_judgment"],
                    )
                    
                    # Build error case
                    error_case = {
                        "fact": result["fact"],
                        "error_reason": error_reason,
                        "true_charges": result["true_charges"],
                        "predicted_charges": result["predicted_charges"],
                        "true_articles": result["true_articles"],
                        "predicted_articles": result["predicted_articles"],
                        "predicted_judgment": result["predicted_judgment"],
                        "predict_prompt_tokens": result["predict_prompt_tokens"],
                        "reason_completion_tokens": result.get("predict_completion_tokens", 0),
                    }
                    
                    error_cases.append(error_case)
                    
                    # Update count
                    for c in result["true_charges"]:
                        if c in charge_count:
                            charge_count[c] += 1
                    
                    # Save immediately
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "metadata": {
                                "mode": "negative_kb_bare_llm",
                                "count": len(error_cases),
                                "total_processed": total_processed,
                                "total_errors": total_errors,
                                "total_correct": total_correct,
                                "per_charge_target": per_charge_target,
                                "charge_count": charge_count,
                                "max_workers": max_workers,
                                "resumed_from": resume_from,
                            },
                            "count": len(error_cases),
                            "error_cases": error_cases,
                        }, f, ensure_ascii=False, indent=2)
                    
                    still_need = [c for c, cnt in charge_count.items() if cnt < per_charge_target]
                    logger.info(f"Saved, {len(still_need)} charges still need more errors")
                    
                    # Check if done
                    if all_charges_done():
                        logger.info("🎉 All charges completed!")
                        done = True
                        break
        
        # Update index and progress bar
        if pbar:
            pbar.update(batch_end - current_idx)
        current_idx = batch_end
        
        # Re-filter after batch: remove any newly completed charges
        # This ensures next batch only processes needed cases
        if not done:
            remaining_data = filter_remaining_cases(remaining_data[current_idx:], charge_count, per_charge_target)
            current_idx = 0
            random.shuffle(remaining_data)
    
    if pbar:
        pbar.close()
    
    # ========== 最终保存 ==========
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "mode": "negative_kb_bare_llm",
                "count": len(error_cases),
                "total_processed": total_processed,
                "total_errors": total_errors,
                "total_correct": total_correct,
                "per_charge_target": per_charge_target,
                "charge_count": charge_count,
                "max_workers": max_workers,
                "resumed_from": resume_from,
                "error_rate": total_errors / total_processed if total_processed > 0 else 0,
                "done_charges": sum(1 for cnt in charge_count.values() if cnt >= per_charge_target),
                "total_charges": total_charges,
            },
            "count": len(error_cases),
            "error_cases": error_cases,
        }, f, ensure_ascii=False, indent=2)
    
    # Final summary
    logger.info("=" * 70)
    logger.info(f"Collection completed!")
    logger.info(f"  Total cases processed: {total_processed}")
    logger.info(f"  Total error cases collected: {len(error_cases)}")
    done_count = sum(1 for cnt in charge_count.values() if cnt >= per_charge_target)
    logger.info(f"  Charges completed: {done_count}/{total_charges} ({per_charge_target} each)")
    incomplete = [(c, cnt) for c, cnt in charge_count.items() if cnt < per_charge_target]
    if incomplete:
        if len(incomplete) <= 20:
            logger.info(f"  Incomplete: {incomplete}")
        else:
            logger.info(f"  Incomplete: {len(incomplete)} charges (first 10: {incomplete[:10]}...)")
    logger.info(f"  Error rate: {total_errors / total_processed * 100:.2f}%" if total_processed > 0 else "N/A")
    logger.info(f"  Parallel workers: {max_workers}")
    logger.info(f"  Saved to: {output_file}")
    logger.info("=" * 70)


def load_api_config(config: dict):
    """Load API configuration from config"""
    api_config = config.get("api", {})
    base_url = api_config.get("base_url")
    api_key_env = api_config.get("api_key_env", "DEEPSEEK_API_KEY")
    model_name = api_config.get("model_name")
    
    api_key = os.getenv(api_key_env)
    
    if not all([base_url, api_key, model_name]):
        raise ValueError(
            f"Missing API configuration:\n"
            f"  - base_url: {base_url}\n"
            f"  - api_key from env '{api_key_env}': {api_key is not None}\n"
            f"  - model_name: {model_name}\n"
            f"Please check config.json and environment variable."
        )
    
    return base_url, api_key, model_name


def main():
    parser = argparse.ArgumentParser(description='Collect error cases for negative knowledge base')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--train-file', type=str, default=None, help='Training data file path')
    parser.add_argument('--per-charge', type=int, default=None, help='Number of error cases per charge')
    parser.add_argument('--output', type=str, default=None, help='Output knowledge base path')
    parser.add_argument('--max-workers', type=int, default=None, help='Parallel workers, adjust by API rate limit')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from existing checkpoint')
    args = parser.parse_args()
    
    # Read config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Load API config
    base_url, api_key, model_name = load_api_config(config)
    
    # Get collection config from config, command line args override
    collect_config = config.get("collection", {})
    per_charge = args.per_charge or collect_config.get("per_charge", 3)
    output = args.output or collect_config.get("output", "data/negative_error_cases/negative_kb_bare_llm.json")
    max_workers = args.max_workers or collect_config.get("max_workers", 10)
    seed = args.seed or collect_config.get("seed", 42)
    
    # Initialize client
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # Load charge and article lists
    charge_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/accu.txt')
    charge_names = []
    if os.path.exists(charge_path):
        with open(charge_path, 'r', encoding='utf-8') as f:
            charge_names = [line.strip() for line in f if line.strip()]
    
    article_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/law.txt')
    article_names = []
    if os.path.exists(article_path):
        with open(article_path, 'r', encoding='utf-8') as f:
            article_names = [line.strip() for line in f if line.strip()]
    
    if not charge_names:
        raise ValueError("Charge list not found at: " + charge_path)
    
    # Load training data
    train_file = args.train_file or collect_config.get("train_file", 'data/final_all_data/first_stage/train.json')
    
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
        charge_names,
        article_names,
        output,
        per_charge_target=per_charge,
        max_workers=max_workers,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
