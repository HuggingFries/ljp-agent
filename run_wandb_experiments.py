"""
Wandb 可视化对比实验脚本
支持多种配置对比消融实验：
- Zero-Shot Baseline
- Flat RAG (固定k vs 自适应k)
- Hierarchical RAG (固定k vs 自适应k)

自动记录：
- 准确率（罪名、法条）
- Micro P/R/F1
- 每个样本预测详情
- 自适应k分布统计
- token消耗统计

Usage:
    pip install wandb
    wandb login
    python run_wandb_experiments.py --test-file data/final_all_data/first_stage/test.json --max-samples 100 --run-all
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import wandb
except ImportError:
    print("⚠️  wandb not installed. Please install first:")
    print("   pip install wandb")
    print("   wandb login")
    exit(1)

from agent import Case, DataLoader, LJPAgentWithRAG, PredictionResult
from baseline import ZeroShotLJPBaseline
from retriever import (
    FlatAdaptiveRAGRetriever,
    HierarchicalAdaptiveRAGRetriever,
    create_retriever_from_config,
)
from run_agent import load_api_config, clean_charge, clean_article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_baseline(
    model: ZeroShotLJPBaseline,
    test_data: List[dict],
    max_samples: Optional[int] = None,
    project_name: str = "ljp-experiments"
) -> Tuple[float, float, float, float, float]:
    """评估Zero-Shot Baseline，记录结果到wandb"""
    run = wandb.init(
        project=project_name,
        name="zero-shot-baseline",
        group="baseline",
        config={
            "method": "zero-shot",
            "retrieval": "none",
            "adaptive_k": False,
        }
    )

    correct_charges = 0
    correct_articles = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # 随机采样
    eval_data = test_data
    if max_samples is not None and max_samples < len(test_data):
        import random
        random.seed(42)
        eval_data = random.sample(test_data, max_samples)

    # 加载标签映射
    charge_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/accu.txt')
    article_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/law.txt')
    charge_names = None
    article_names = None
    if os.path.exists(charge_path):
        with open(charge_path, 'r', encoding='utf-8') as f:
            charge_names = [line.strip() for line in f if line.strip()]
            model.charge_names = charge_names
    if os.path.exists(article_path):
        with open(article_path, 'r', encoding='utf-8') as f:
            article_names = [line.strip() for line in f if line.strip()]
            model.article_names = article_names

    sample_table = wandb.Table(columns=[
        "index", "fact", "true_charges", "predicted_charges",
        "true_articles", "predicted_articles", "correct_charge", "correct_article"
    ])

    # 并发锁保护全局变量
    lock = Lock()
    results = []

    def process_sample(i, item):
        """单个样本处理函数，给并发池调用"""
        case = DataLoader.convert_to_case(item, is_positive=False)
        true_charges_set = set(map(clean_charge, case.charges))
        true_articles_set = set(map(clean_article, case.articles))

        try:
            result = model.predict(case)
            pred_charges = set(map(clean_charge, result.predicted_charges))
            pred_articles = set(map(clean_article, result.predicted_articles))

            charge_correct = pred_charges == true_charges_set
            article_correct = pred_articles == true_articles_set

            return {
                "i": i,
                "case": case,
                "true_charges_set": true_charges_set,
                "true_articles_set": true_articles_set,
                "pred_charges": pred_charges,
                "pred_articles": pred_articles,
                "charge_correct": charge_correct,
                "article_correct": article_correct,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error on baseline sample {i}: {e}", exc_info=True)
            return {
                "i": i,
                "error": str(e)
            }

    # 并发执行，默认最多8个并发（避免限流）
    max_workers = min(8, len(eval_data))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, i, item) for i, item in enumerate(eval_data)]
        for future in tqdm(futures, desc="Evaluating baseline (concurrent)"):
            res = future.result()
            if res.get("error") is not None:
                continue

            i = res["i"]
            case = res["case"]
            true_charges_set = res["true_charges_set"]
            true_articles_set = res["true_articles_set"]
            pred_charges = res["pred_charges"]
            pred_articles = res["pred_articles"]
            charge_correct = res["charge_correct"]
            article_correct = res["article_correct"]

            # 加锁更新全局统计
            with lock:
                # 统计TP/FP/FN
                for c in pred_charges:
                    if c in true_charges_set:
                        tp += 1
                    else:
                        fp += 1
                for c in true_charges_set:
                    if c not in pred_charges:
                        fn += 1

                if charge_correct:
                    correct_charges += 1
                if article_correct:
                    correct_articles += 1

                total_prompt_tokens += res["prompt_tokens"]
                total_completion_tokens += res["completion_tokens"]
                total += 1

                # 添加到表格
                sample_table.add_data(
                    i,
                    case.fact[:500] + ("..." if len(case.fact) > 500 else ""),
                    ", ".join(true_charges_set),
                    ", ".join(pred_charges),
                    ", ".join(true_articles_set),
                    ", ".join(pred_articles),
                    charge_correct,
                    article_correct
                )

    # 计算指标
    acc_charge = correct_charges / total if total > 0 else 0.0
    acc_article = correct_articles / total if total > 0 else 0.0

    if total == 0:
        # 没有成功预测的样本，所有指标归零
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

    # 记录到wandb
    wandb.log({
        "charge_accuracy": acc_charge,
        "article_accuracy": acc_article,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "total_samples": total,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_prompt_tokens_per_sample": total_prompt_tokens / total if total > 0 else 0,
        "samples_table": sample_table
    })

    logger.info(f"Baseline done: Charge Acc={acc_charge:.4f}, Article Acc={acc_article:.4f}, Micro F1={f1:.4f}")
    wandb.finish()

    return acc_charge, acc_article, precision, recall, f1


def evaluate_rag(
    config: dict,
    base_url: str,
    api_key: str,
    model_name: str,
    test_data: List[dict],
    charge_names: List[str],
    article_names: List[str],
    embedding_model: SentenceTransformer,
    max_samples: Optional[int],
    project_name: str,
    run_name: str,
    group_name: str,
    adaptive_k: bool,
    retriever_mode: str,
    k_positive: Optional[int] = None,
    k_negative: Optional[int] = None,
) -> Tuple[float, float, float, float, float]:
    """评估RAG方法，记录到wandb"""
    # 初始化wandb
    run = wandb.init(
        project=project_name,
        name=run_name,
        group=group_name,
        config={
            "method": "rag",
            "retrieval": retriever_mode,
            "adaptive_k": adaptive_k,
            "k_positive": k_positive,
            "k_negative": k_negative,
        }
    )

    # 配置修改为当前检索模式
    config_retriever = config.get("retriever", {})
    config_retriever["mode"] = retriever_mode
    config["retriever"] = config_retriever

    # 创建OpenAI客户端给检索器
    import os
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)

    # 创建检索器
    retriever = create_retriever_from_config(config["retriever"], client, model_name)

    # 创建Agent
    agent = LJPAgentWithRAG(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        adaptive_retriever=retriever,
        k_positive=k_positive if not adaptive_k else None,
        k_negative=k_negative if not adaptive_k else None,
        charge_names=charge_names,
        article_names=article_names
    )

    # 随机采样
    eval_data = test_data
    if max_samples is not None and max_samples < len(test_data):
        import random
        random.seed(42)
        eval_data = random.sample(test_data, max_samples)

    correct_charges = 0
    correct_articles = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # 收集自适应k分布
    adaptive_k_pos_list = []
    adaptive_k_neg_list = []
    max_sim_list = []

    sample_table = wandb.Table(columns=[
        "index", "fact", "true_charges", "predicted_charges",
        "true_articles", "predicted_articles", "k_pos", "k_neg", "max_sim",
        "correct_charge", "correct_article"
    ])

    # 并发锁保护全局变量
    lock = Lock()

    def process_sample(i, item):
        """单个样本处理函数，给并发池调用"""
        case = DataLoader.convert_to_case(item, is_positive=False)
        true_charges_set = set(map(clean_charge, case.charges))
        true_articles_set = set(map(clean_article, case.articles))

        try:
            result = agent.predict(case, embedding_model)
            pred_charges = set(map(clean_charge, result.predicted_charges))
            pred_articles = set(map(clean_article, result.predicted_articles))

            charge_correct = pred_charges == true_charges_set
            article_correct = pred_articles == true_articles_set

            # 收集自适应信息
            k_pos_this = k_positive
            k_neg_this = k_negative
            max_sim = None
            if adaptive_k and hasattr(result, 'retrieval_info'):
                k_pos_this = result.retrieval_info.positive_k
                k_neg_this = result.retrieval_info.negative_k
                max_sim = float(result.retrieval_info.max_sim_pos)

            return {
                "i": i,
                "case": case,
                "true_charges_set": true_charges_set,
                "true_articles_set": true_articles_set,
                "pred_charges": pred_charges,
                "pred_articles": pred_articles,
                "charge_correct": charge_correct,
                "article_correct": article_correct,
                "k_pos_this": k_pos_this,
                "k_neg_this": k_neg_this,
                "max_sim": max_sim,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error on rag sample {i}: {e}", exc_info=True)
            return {
                "i": i,
                "error": str(e)
            }

    # 并发执行，默认最多8个并发（避免限流，可调整）
    max_workers = min(8, len(eval_data))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, i, item) for i, item in enumerate(eval_data)]
        for future in tqdm(futures, desc=f"Evaluating {run_name} (concurrent, {max_workers} workers)"):
            res = future.result()
            if res.get("error") is not None:
                continue

            i = res["i"]
            case = res["case"]
            true_charges_set = res["true_charges_set"]
            true_articles_set = res["true_articles_set"]
            pred_charges = res["pred_charges"]
            pred_articles = res["pred_articles"]
            charge_correct = res["charge_correct"]
            article_correct = res["article_correct"]
            k_pos_this = res["k_pos_this"]
            k_neg_this = res["k_neg_this"]
            max_sim = res["max_sim"]

            # 加锁更新全局统计
            with lock:
                # 统计TP/FP/FN
                for c in pred_charges:
                    if c in true_charges_set:
                        tp += 1
                    else:
                        fp += 1
                for c in true_charges_set:
                    if c not in pred_charges:
                        fn += 1

                if charge_correct:
                    correct_charges += 1
                if article_correct:
                    correct_articles += 1

                total_prompt_tokens += res["prompt_tokens"]
                total_completion_tokens += res["completion_tokens"]
                total += 1

                # 记录自适应k分布
                if adaptive_k and max_sim is not None:
                    adaptive_k_pos_list.append(k_pos_this)
                    adaptive_k_neg_list.append(k_neg_this)
                    max_sim_list.append(max_sim)

                # 添加到表格
                sample_table.add_data(
                    i,
                    case.fact[:500] + ("..." if len(case.fact) > 500 else ""),
                    ", ".join(true_charges_set),
                    ", ".join(pred_charges),
                    ", ".join(true_articles_set),
                    ", ".join(pred_articles),
                    k_pos_this,
                    k_neg_this,
                    max_sim if max_sim is not None else float('nan'),
                    charge_correct,
                    article_correct
                )

    # 计算指标
    acc_charge = correct_charges / total if total > 0 else 0
    acc_article = correct_articles / total if total > 0 else 0

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = tp / (tp + fn)
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # 记录指标
    log_data = {
        "charge_accuracy": acc_charge,
        "article_accuracy": acc_article,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "total_samples": total,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_prompt_tokens_per_sample": total_prompt_tokens / total if total > 0 else 0,
        "samples_table": sample_table
    }

    # 如果是自适应k，记录分布直方图
    if adaptive_k and len(adaptive_k_pos_list) > 0:
        wandb.log({
            "histogram_k_pos": wandb.Histogram(adaptive_k_pos_list),
            "histogram_k_neg": wandb.Histogram(adaptive_k_neg_list),
            "histogram_max_sim": wandb.Histogram(max_sim_list),
        })

    wandb.log(log_data)

    logger.info(f"{run_name} done: Charge Acc={acc_charge:.4f}, Article Acc={acc_article:.4f}, Micro F1={f1:.4f}")
    wandb.finish()

    return acc_charge, acc_article, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='Run LJP experiments with wandb visualization')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--test-file', type=str, default='data/final_all_data/first_stage/test.json',
                       help='Test data path')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum samples per experiment')
    parser.add_argument('--project-name', type=str, default='ljp-rag-experiments',
                       help='Wandb project name')
    parser.add_argument('--run-baseline', action='store_true',
                       help='Run baseline experiment')
    parser.add_argument('--run-flat-fixed', action='store_true',
                       help='Run flat RAG with fixed k')
    parser.add_argument('--run-flat-adaptive', action='store_true',
                       help='Run flat RAG with adaptive k')
    parser.add_argument('--run-hierarchical-fixed', action='store_true',
                       help='Run hierarchical RAG with fixed k')
    parser.add_argument('--run-hierarchical-adaptive', action='store_true',
                       help='Run hierarchical RAG with adaptive k')
    parser.add_argument('--run-all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--fixed-k-pos', type=int, default=3,
                       help='Fixed k for positive examples when running fixed experiments')
    parser.add_argument('--fixed-k-neg', type=int, default=3,
                       help='Fixed k for negative examples when running fixed experiments')
    args = parser.parse_args()

    # 如果--run-all，打开所有
    if args.run_all:
        args.run_baseline = True
        args.run_flat_fixed = True
        args.run_flat_adaptive = True
        args.run_hierarchical_fixed = True
        args.run_hierarchical_adaptive = True

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    base_url, api_key, model_name = load_api_config(config)

    # 加载测试数据
    test_data = DataLoader.load_cail2018(args.test_file)
    logger.info(f"Loaded {len(test_data)} total test samples")

    # 加载标签映射
    charge_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/accu.txt')
    article_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/law.txt')
    charge_names = None
    article_names = None
    if os.path.exists(charge_path):
        with open(charge_path, 'r', encoding='utf-8') as f:
            charge_names = [line.strip() for line in f if line.strip()]
    if os.path.exists(article_path):
        with open(article_path, 'r', encoding='utf-8') as f:
            article_names = [line.strip() for line in f if line.strip()]

    # 1. 跑Baseline
    if args.run_baseline:
        logger.info("=== Starting Zero-Shot Baseline experiment ===")
        model = ZeroShotLJPBaseline(base_url=base_url, api_key=api_key, model_name=model_name)
        evaluate_baseline(model, test_data, args.max_samples, args.project_name)

    # 对于RAG实验，需要加载embedding模型
    need_rag = any([args.run_flat_fixed, args.run_flat_adaptive, args.run_hierarchical_fixed, args.run_hierarchical_adaptive])
    if need_rag:
        logger.info("Loading embedding model...")
        embedding_model_name = config.get("index", {}).get("embedding_model", "uer/sbert-base-chinese-nli")
        embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"Loaded embedding model: {embedding_model_name}")

        # 2. Flat RAG + fixed k (customized)
        if args.run_flat_fixed:
            kp = args.fixed_k_pos
            kn = args.fixed_k_neg
            logger.info(f"=== Starting Flat RAG (fixed k={kp}+{kn}) experiment ===")
            config["retriever"]["mode"] = "flat"
            evaluate_rag(
                config, base_url, api_key, model_name,
                test_data, charge_names, article_names, embedding_model,
                args.max_samples, args.project_name,
                run_name=f"flat-fixed-k-{kp}-{kn}",
                group_name="flat",
                adaptive_k=False,
                retriever_mode="flat",
                k_positive=kp,
                k_negative=kn
            )

        # 3. Flat RAG + adaptive k
        if args.run_flat_adaptive:
            logger.info("=== Starting Flat RAG (adaptive k) experiment ===")
            config["retriever"]["mode"] = "flat"
            evaluate_rag(
                config, base_url, api_key, model_name,
                test_data, charge_names, article_names, embedding_model,
                args.max_samples, args.project_name,
                run_name="flat-adaptive-k",
                group_name="flat",
                adaptive_k=True,
                retriever_mode="flat"
            )

        # 4. Hierarchical RAG + fixed k (customized)
        if args.run_hierarchical_fixed:
            kp = args.fixed_k_pos
            kn = args.fixed_k_neg
            logger.info(f"=== Starting Hierarchical RAG (fixed k={kp}+{kn}) experiment ===")
            config["retriever"]["mode"] = "hierarchical"
            evaluate_rag(
                config, base_url, api_key, model_name,
                test_data, charge_names, article_names, embedding_model,
                args.max_samples, args.project_name,
                run_name=f"hierarchical-fixed-k-{kp}-{kn}",
                group_name="hierarchical",
                adaptive_k=False,
                retriever_mode="hierarchical",
                k_positive=kp,
                k_negative=kn
            )

        # 5. Hierarchical RAG + adaptive k
        if args.run_hierarchical_adaptive:
            logger.info("=== Starting Hierarchical RAG (adaptive k) experiment ===")
            config["retriever"]["mode"] = "hierarchical"
            evaluate_rag(
                config, base_url, api_key, model_name,
                test_data, charge_names, article_names, embedding_model,
                args.max_samples, args.project_name,
                run_name="hierarchical-adaptive-k",
                group_name="hierarchical",
                adaptive_k=True,
                retriever_mode="hierarchical"
            )

    # 汇总所有结果到总结表格
    logger.info("All experiments completed! Check results in your wandb project.")


if __name__ == "__main__":
    main()
