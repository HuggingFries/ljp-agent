#!/usr/bin/env python3
"""
断点续收集裸LLM错误案例
从已有的收集结果加载进度，过滤出仅包含未填满罪名的测试集，继续收集
解决越跑越慢问题：后续只处理需要的案件，不会空跑

Author: LJP Project
Date: 2026-04-13
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
    """清洗罪名：去掉末尾'罪'字"""
    charge = charge.strip()
    if charge.endswith("罪"):
        charge = charge[:-1]
    return charge


def clean_article(article: str) -> str:
    """清洗法条：只保留数字"""
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
    """裸LLM直接预测，不使用任何RAG检索"""
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
) -> str:
    """让LLM提取抽象错误原因"""
    prompt = f"""请你作为法律AI专家，分析这个判决预测错误的案例，提取抽象、概括的错误原因。

## 案件事实
{fact}

## 真实正确罪名
{', '.join(true_charges)}

## AI预测错误罪名
{', '.join(pred_charges)}

## 任务
请你提炼出1-3句话抽象概括错误原因，重点关注：
1. 为什么模型会预测错误？是哪些事实特征导致混淆？
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
    """加载CAIL2018格式数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} cases from {file_path}")
    return data


def filter_test_data_by_charge(
    test_data: List[dict],
    charge_count: Dict[str, int],
    per_charge_target: int,
) -> List[dict]:
    """过滤测试集：只保留至少包含一个未填满罪名的案件
    这样后续处理不会空跑，每个案件都有处理必要，解决越跑越慢问题
    """
    filtered = []
    for item in test_data:
        charges = item.get('charge', [])
        if not charges and 'meta' in item:
            charges = item['meta'].get('accusation', [])
        true_charges = list(map(clean_charge, charges))
        # 只要有一个罪名没满，就保留
        for c in true_charges:
            if charge_count.get(c, 0) < per_charge_target:
                filtered.append(item)
                break
    logger.info(f"过滤完成：原 {len(test_data)} 个案件 → 过滤后 {len(filtered)} 个需要处理")
    return filtered


def process_single_case(
    item: dict,
    client: OpenAI,
    model_name: str,
    charge_names: List[str],
    article_names: List[str],
    charge_count: Dict[str, int],
    per_charge_target: int,
) -> Dict[str, Any]:
    """并行处理单个案件：预测 + 判断是否需要收集"""
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
    
    # 已经过滤过了，这里肯定有罪名没满，直接处理
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
            return {
                "type": "correct",
            }
        
        # 预测错误，需要收集
        return {
            "type": "error_needed",
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
            "type": "error",
            "error": str(e),
        }


def resume_collect(
    client: OpenAI,
    model_name: str,
    original_test_data: List[dict],
    existing_result_path: str,
    charge_names: List[str],
    article_names: List[str],
    output_file: str,
    per_charge_target: int = 3,
    max_workers: int = 10,
) -> None:
    """断点续收集：从已有结果继续收集"""
    
    # ========== 第一步：加载已有收集结果 ==========
    logger.info(f"加载已有收集结果: {existing_result_path}")
    with open(existing_result_path, 'r', encoding='utf-8') as f:
        existing = json.load(f)
    
    existing_error_cases = existing.get("error_cases", [])
    existing_charge_count = existing.get("metadata", {}).get("charge_count", {})
    existing_total_processed = existing.get("metadata", {}).get("total_processed", 0)
    existing_total_errors = existing.get("metadata", {}).get("total_errors", 0)
    existing_total_correct = existing.get("metadata", {}).get("total_correct", 0)
    
    logger.info(f"已有结果: {len(existing_error_cases)} 错误案例，进度:")
    done = sum(1 for cnt in existing_charge_count.values() if cnt >= per_charge_target)
    total = len(existing_charge_count)
    logger.info(f"  {done}/{total} 个罪名已完成，{total - done} 个仍需收集")
    
    # ========== 第二步：过滤测试集，只保留需要处理的案件 ==========
    logger.info("过滤测试集，只保留包含未填满罪名的案件...")
    filtered_test_data = filter_test_data_by_charge(
        original_test_data,
        existing_charge_count,
        per_charge_target,
    )
    
    if not filtered_test_data:
        logger.info("所有罪名都已收集完成！不需要继续处理了")
        return
    
    # ========== 第三步：打乱顺序，开始并行收集 ==========
    charge_count = existing_charge_count.copy()
    error_cases = existing_error_cases.copy()
    total_processed = existing_total_processed
    total_errors = existing_total_errors
    total_correct = existing_total_correct
    
    total_charges = len(charge_count)
    logger.info(f"[断点续收集] 并发线程数: {max_workers}，继续收集...")
    
    random.shuffle(filtered_test_data)
    
    # 检查是否已全部完成
    def all_charges_done():
        return all(cnt >= per_charge_target for cnt in charge_count.values())
    
    batch_size = max_workers * 5
    done = False
    
    if has_tqdm:
        pbar = tqdm(total=len(filtered_test_data), desc="Remaining cases processed")
    else:
        pbar = None
    
    i = 0
    while i < len(filtered_test_data) and not done:
        batch_end = min(i + batch_size, len(filtered_test_data))
        batch = filtered_test_data[i:batch_end]
        
        logger.info(f"处理批次: {len(batch)} 个剩余案件 (已处理 {i}/{len(filtered_test_data)})")
        
        # 并行处理
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
                    charge_count,
                    per_charge_target,
                )
                futures.append(future)
            
            for future in as_completed(futures):
                total_processed += 1
                result = future.result()
                
                if result is None:
                    continue
                
                if result["type"] == "correct":
                    total_correct += 1
                    pass
                elif result["type"] == "error_needed":
                    total_errors += 1
                    logger.info(f"发现新错误案例：真实={sorted(result['true_charges'])}, 预测={sorted(result['predicted_charges'])}")
                    
                    error_reason = extract_error_reason(
                        client,
                        model_name,
                        result["fact"],
                        result["true_charges"],
                        result["predicted_charges"],
                    )
                    
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
                    
                    # 更新计数
                    for c in result["true_charges"]:
                        if c in charge_count:
                            charge_count[c] += 1
                    
                    # 即时保存
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "count": len(error_cases),
                            "error_cases": error_cases,
                            "metadata": {
                                "mode": "bare_llm_parallel_resumed",
                                "total_processed": total_processed,
                                "total_errors": total_errors,
                                "total_correct": total_correct,
                                "per_charge_target": per_charge_target,
                                "charge_count": charge_count,
                                "max_workers": max_workers,
                                "resumed_from": existing_result_path,
                            }
                        }, f, ensure_ascii=False, indent=2)
                    
                    still_need = [c for c, cnt in charge_count.items() if cnt < per_charge_target]
                    logger.info(f"已保存，剩余 {len(still_need)} 个罪名还需要收集")
                    
                    if all_charges_done():
                        logger.info("🎉 所有罪名都已收集完成！")
                        done = True
                        break
                
                elif result["type"] == "error":
                    pass
        
        i = batch_end
        if pbar:
            pbar.update(batch_end - i)
        
        if done:
            break
    
    if pbar:
        pbar.close()
    
    # 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "count": len(error_cases),
            "error_cases": error_cases,
            "metadata": {
                "mode": "bare_llm_parallel_resumed",
                "total_processed": total_processed,
                "total_errors": total_errors,
                "total_correct": total_correct,
                "per_charge_target": per_charge_target,
                "charge_count": charge_count,
                "max_workers": max_workers,
                "resumed_from": existing_result_path,
                "error_rate": total_errors / total_processed if total_processed > 0 else 0,
                "done_charges": sum(1 for cnt in charge_count.values() if cnt >= per_charge_target),
                "total_charges": total_charges,
            }
        }, f, ensure_ascii=False, indent=2)
    
    # 输出统计
    logger.info("=" * 70)
    logger.info(f"断点续收集完成！")
    logger.info(f"  累计处理案件: {total_processed}")
    logger.info(f"  累计错误案例: {len(error_cases)}")
    done_count = sum(1 for cnt in charge_count.values() if cnt >= per_charge_target)
    logger.info(f"  完成罪名: {done_count}/{total_charges} (每个{per_charge_target}个)")
    incomplete = [(c, cnt) for c, cnt in charge_count.items() if cnt < per_charge_target]
    if incomplete:
        if len(incomplete) <= 20:
            logger.info(f"  未完成: {incomplete}")
        else:
            logger.info(f"  未完成: {len(incomplete)} 个罪名 (其中: {incomplete[:10]}...)")
    logger.info(f"  错误率: {total_errors / total_processed * 100:.2f}%" if total_processed > 0 else "N/A")
    logger.info(f"  并发线程数: {max_workers}")
    logger.info(f"  保存位置: {output_file}")
    logger.info("=" * 70)


def load_api_config(config: dict):
    """加载API配置"""
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
    parser = argparse.ArgumentParser(description='[断点续收集] 继续收集裸LLM判错案例')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--from', dest='existing_result', required=True, help='已有的收集结果json文件（继续收集的起点）')
    parser.add_argument('--test-file', type=str, default=None, help='原测试数据文件路径，覆盖配置')
    parser.add_argument('--per-charge', type=int, default=3, help='每个罪名目标多少个错误')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径，默认覆盖原文件')
    parser.add_argument('--max-workers', type=int, default=10, help='并发线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 加载API配置
    base_url, api_key, model_name = load_api_config(config)
    
    # 初始化客户端
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # 加载标签映射（所有罪名列表）
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
        raise ValueError("找不到罪名列表，请检查文件: " + charge_path)
    
    # 加载原始测试数据
    eval_config = config.get("evaluation", {})
    test_file = args.test_file or eval_config.get("test_file", 'data/final_all_data/first_stage/test.json')
    
    random.seed(args.seed)
    original_test_data = load_cail2018(test_file)
    logger.info(f"加载原始测试数据完成，共 {len(original_test_data)} 个案件")
    
    # 输出文件默认覆盖原文件
    output_file = args.output or args.existing_result
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 开始断点续收集
    resume_collect(
        client,
        model_name,
        original_test_data,
        args.existing_result,
        charge_names,
        article_names,
        output_file,
        per_charge_target=args.per_charge,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()