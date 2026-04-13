#!/usr/bin/env python3
"""
收集裸LLM判错案例脚本 - 并行版本
使用多线程并发调用API，大幅提高速度

特点：
1. 直接调用裸LLM，不使用RAG
2. 每个罪名收集固定数量错误
3. 多线程并行API调用，加速收集
4. 先判断错误再提取原因，不浪费token

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


def process_single_case(
    item: dict,
    client: OpenAI,
    model_name: str,
    charge_names: List[str],
    article_names: List[str],
    charge_count: Dict[str, int],
    per_charge_target: int,
) -> Dict[str, Any]:
    """并行处理单个案件：预测 + 判断是否需要收集
    
    Returns:
        如果是需要收集的错误案例，返回完整数据；否则返回None
    """
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
    
    # 检查：是否有罪名还没收够
    need_this_case = False
    for c in true_charges:
        if charge_count.get(c, 0) < per_charge_target:
            need_this_case = True
            break
    
    if not need_this_case:
        return None  # 不需要这个案例
    
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
            return {
                "type": "correct",
                "fact": fact,
                "true_charges": list(true_charges_set),
            }
        
        # 预测错误，且需要这个案例 -> 返回数据待后续提取错误原因
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
            "fact": fact,
            "error": str(e),
        }


def collect_error_cases_parallel(
    client: OpenAI,
    model_name: str,
    test_data: List[dict],
    charge_names: List[str],
    article_names: List[str],
    output_file: str,
    per_charge_target: int = 3,
    max_workers: int = 10,
) -> None:
    """并行收集裸LLM预测错误案例
    
    Args:
        max_workers: 线程池大小，并发调用数量，根据API限制调整
    """
    error_cases: List[Dict[str, Any]] = []
    total_processed = 0
    total_errors = 0
    total_correct = 0
    
    # 初始化每个罪名计数
    charge_count: Dict[str, int] = {}
    for c in charge_names:
        charge_count[clean_charge(c)] = 0
    
    total_charges = len(charge_count)
    logger.info(f"[并行版本] 按罪名固定收集裸LLM错误：每个罪名目标 {per_charge_target} 个，共 {total_charges} 个罪名")
    logger.info(f"并发线程数: {max_workers}，预计总共收集 ~{total_charges * per_charge_target} 个错误")
    
    # 打乱测试集顺序
    random.shuffle(test_data)
    
    # 检查是否已全部完成
    def all_charges_done():
        return all(cnt >= per_charge_target for cnt in charge_count.values())
    
    # 批量处理：每次取一批案件并行处理
    batch_size = max_workers * 5
    done = False
    
    if has_tqdm:
        pbar = tqdm(total=len(test_data), desc="Total cases processed")
    else:
        pbar = None
    
    i = 0
    while i < len(test_data) and not done:
        # 取一批案件
        batch_end = min(i + batch_size, len(test_data))
        batch = test_data[i:batch_end]
        
        # 过滤掉：所有罪名都已满的案件（不处理，节省API）
        filtered_batch = []
        for item in batch:
            # 快速检查：是否至少有一个罪名还没收满
            charges = item.get('charge', [])
            if not charges and 'meta' in item:
                charges = item['meta'].get('accusation', [])
            true_charges = list(map(clean_charge, charges))
            for c in true_charges:
                if charge_count.get(c, 0) < per_charge_target:
                    filtered_batch.append(item)
                    break
        
        if not filtered_batch:
            i = batch_end
            if pbar:
                pbar.update(batch_end - i)
            continue
        
        logger.info(f"处理批次: {len(filtered_batch)} 个案件 (已处理 {i}/{len(test_data)})")
        
        # 并行处理批次中每个案件
        with ThreadPoolExecutor(max_workers=min(max_workers, len(filtered_batch))) as executor:
            futures = []
            for item in filtered_batch:
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
            
            # 收集结果
            for future in as_completed(futures):
                total_processed += 1
                result = future.result()
                
                if result is None:
                    continue
                
                if result["type"] == "correct":
                    total_correct += 1
                    # 正确案件不需要，但是对应罪名还是没满，继续找
                    pass
                elif result["type"] == "error_needed":
                    # 预测错误，且需要这个案例 -> 提取错误原因
                    total_errors += 1
                    logger.info(f"发现需要的错误案例：真实={sorted(result['true_charges'])}, 预测={sorted(result['predicted_charges'])}")
                    
                    error_reason = extract_error_reason(
                        client,
                        model_name,
                        result["fact"],
                        result["true_charges"],
                        result["predicted_charges"],
                    )
                    
                    # 组装完整错误案例
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
                                "mode": "bare_llm_parallel",
                                "total_processed": total_processed,
                                "total_errors": total_errors,
                                "total_correct": total_correct,
                                "per_charge_target": per_charge_target,
                                "charge_count": charge_count,
                                "max_workers": max_workers,
                            }
                        }, f, ensure_ascii=False, indent=2)
                    
                    still_need = [c for c, cnt in charge_count.items() if cnt < per_charge_target]
                    logger.info(f"已保存，剩余 {len(still_need)} 个罪名还需要收集")
                    
                    # 检查是否全部完成
                    if all_charges_done():
                        logger.info("所有罪名都已收集完成！")
                        done = True
                        break
                
                elif result["type"] == "error":
                    # 处理出错，跳过
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
                "mode": "bare_llm_parallel",
                "total_processed": total_processed,
                "total_errors": total_errors,
                "total_correct": total_correct,
                "per_charge_target": per_charge_target,
                "charge_count": charge_count,
                "max_workers": max_workers,
                "error_rate": total_errors / total_processed if total_processed > 0 else 0,
                "done_charges": sum(1 for cnt in charge_count.values() if cnt >= per_charge_target),
                "total_charges": total_charges,
            }
        }, f, ensure_ascii=False, indent=2)
    
    # 输出统计
    logger.info("=" * 70)
    logger.info(f"收集完成！")
    logger.info(f"  处理案件总数: {total_processed}")
    logger.info(f"  收集错误案例: {len(error_cases)}")
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
    parser = argparse.ArgumentParser(description='[并行版] 收集裸LLM判错案例到负例知识库（每个罪名固定收集）')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--test-file', type=str, default=None, help='测试数据文件路径，覆盖配置')
    parser.add_argument('--per-charge', type=int, default=3, help='每个罪名收集多少个错误案例')
    parser.add_argument('--output', type=str, default='data/negative_error_cases/bare_llm_error_kb_parallel.json', help='输出文件路径')
    parser.add_argument('--max-workers', type=int, default=10, help='并发线程数，根据API限流调整')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（打乱测试集顺序）')
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
    
    # 加载测试数据
    eval_config = config.get("evaluation", {})
    test_file = args.test_file or eval_config.get("test_file", 'data/final_all_data/first_stage/test.json')
    
    random.seed(args.seed)
    test_data_full = load_cail2018(test_file)
    logger.info(f"加载测试数据完成，共 {len(test_data_full)} 个案件")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 开始并行收集
    collect_error_cases_parallel(
        client,
        model_name,
        test_data_full,
        charge_names,
        article_names,
        args.output,
        per_charge_target=args.per_charge,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()