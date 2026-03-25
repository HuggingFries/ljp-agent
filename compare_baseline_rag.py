"""
对比Zero-Shot Baseline 和 RAG+正负案例 在同一个随机测试子集上的性能
保证采样相同，公平对比
"""

import os
import json
import random
import argparse
import logging
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from agent import Case, DataLoader, LJPAgentWithRAG
from baseline import ZeroShotLJPBaseline
from retriever import EmbeddingRetriever
from main import load_api_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_charge(charge: str) -> str:
    charge = charge.strip()
    if charge.endswith("罪"):
        charge = charge[:-1]
    return charge


def clean_article(article: str) -> str:
    article = article.strip()
    digits = ''.join([c for c in article if c.isdigit()])
    if digits:
        return digits
    article = article.replace("《中华人民共和国刑法》", "").replace("第", "").replace("条", "").replace("款", "").strip()
    digits = ''.join([c for c in article if c.isdigit()])
    return digits if digits else article


def evaluate_model(
    model,
    test_data: List[dict],
    charge_names: List[str],
    article_names: List[str],
    embedding_model=None,
) -> Tuple[float, float, float, float, float, List[dict]]:
    """通用评估函数，兼容baseline和rag"""
    correct_charges = 0
    correct_articles = 0
    total = 0
    details = []
    
    tp = 0
    fp = 0
    fn = 0
    
    total_samples = len(test_data)
    for i, item in enumerate(test_data):
        case = DataLoader.convert_to_case(item, is_positive=False)
        
        true_charges_set = set(map(clean_charge, case.charges))
        true_articles_set = set(map(clean_article, case.articles))
        
        detail = {
            "index": i,
            "fact": case.fact,
            "true_charges": list(true_charges_set),
            "true_articles": list(true_articles_set),
        }
        
        # 每个样本都标上序号，方便随时检查进度
        logger.info(f"[{i+1}/{total_samples}] Processing sample...")
        
        try:
            # 判断是不是RAG模型：看predict方法参数
            import inspect
            sig = inspect.signature(model.predict)
            if 'embedding_model' in sig.parameters:
                # RAG model 需要embedding_model
                result = model.predict(case, embedding_model)
            else:
                # baseline 不需要
                result = model.predict(case)
            
            pred_charges = set(map(clean_charge, result.predicted_charges))
            pred_articles = set(map(clean_article, result.predicted_articles))
            
            detail["predicted_charges"] = list(pred_charges)
            detail["predicted_articles"] = list(pred_articles)
            detail["predicted_judgment"] = result.predicted_judgment
            
            # 如果是自适应RAG，记录自适应k信息
            if 'retrieval_info' in dir(result):
                detail["adaptive_k"] = {
                    "positive_k": result.retrieval_info.positive_k,
                    "negative_k": result.retrieval_info.negative_k,
                    "max_sim_pos": result.retrieval_info.max_sim_pos,
                    "max_sim_neg": result.retrieval_info.max_sim_neg
                }
            
            # 统计TP/FP/FN
            for c in pred_charges:
                if c in true_charges_set:
                    tp += 1
                else:
                    fp += 1
            for c in true_charges_set:
                if c not in pred_charges:
                    fn += 1
            
            charge_correct = pred_charges == true_charges_set
            article_correct = pred_articles == true_articles_set
            
            if charge_correct:
                correct_charges += 1
            if article_correct:
                correct_articles += 1
            
            detail["correct_charge"] = charge_correct
            detail["correct_article"] = article_correct
            total += 1
        
        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            detail["error"] = str(e)
        
        if 'baseline' in dir():
            # 我们现在是在分开填充baseline和rag的结果
            pass
        
        details.append(detail)
    
    acc_charge = correct_charges / total if total > 0 else 0
    acc_article = correct_articles / total if total > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return acc_charge, acc_article, precision, recall, f1, details


def load_label_mappings(data_dir: str = "data/cail2018/baseline"):
    charge_path = os.path.join(data_dir, "accu.txt")
    article_path = os.path.join(data_dir, "law.txt")
    
    charge_names = []
    if os.path.exists(charge_path):
        with open(charge_path, 'r', encoding='utf-8') as f:
            charge_names = [line.strip() for line in f if line.strip()]
    
    article_names = []
    if os.path.exists(article_path):
        with open(article_path, 'r', encoding='utf-8') as f:
            article_names = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(charge_names)} charges, {len(article_names)} articles")
    return charge_names, article_names


def load_rag_index(
    index_dir: str = "data",
    embedding_model_name: str = "uer/sbert-base-chinese-nli",
    min_k: int = 1,
    max_k: int = 5,
    alpha: float = 1.0,
    normalize: bool = False,
    sim_min: float = None,
    sim_max: float = None,
) -> Tuple[AdaptiveRAGRetriever, SentenceTransformer]:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    embedding_model = SentenceTransformer(embedding_model_name)
    
    pos_cases_path = os.path.join(index_dir, "pos_cases.json")
    pos_index_path = os.path.join(index_dir, "pos_index.npy")
    
    with open(pos_cases_path, 'r', encoding='utf-8') as f:
        pos_cases_data = json.load(f)
    
    pos_cases = []
    for item in pos_cases_data:
        pos_cases.append(Case(
            fact=item['fact'],
            charges=item['charges'],
            articles=item['articles'],
            judgment='',
            is_positive=True
        ))
    
    pos_embeddings = np.load(pos_index_path)
    pos_retriever = EmbeddingRetriever(
        embedding_model=None,
        min_k=min_k, 
        max_k=max_k, 
        alpha=alpha,
        normalize=normalize
    )
    pos_retriever.index(pos_cases, pos_embeddings)
    
    neg_cases_path = os.path.join(index_dir, "neg_cases.json")
    neg_index_path = os.path.join(index_dir, "neg_index.npy")
    
    with open(neg_cases_path, 'r', encoding='utf-8') as f:
        neg_cases_data = json.load(f)
    
    neg_cases = []
    for item in neg_cases_data:
        neg_cases.append(Case(
            fact=item['fact'],
            charges=item['charges'],
            articles=item['articles'],
            judgment='',
            is_positive=False
        ))
    
    neg_embeddings = np.load(neg_index_path)
    neg_retriever = EmbeddingRetriever(
        embedding_model=None,
        min_k=min_k, 
        max_k=max_k, 
        alpha=alpha,
        normalize=normalize
    )
    neg_retriever.index(neg_cases, neg_embeddings)
    
    # 如果开启normalize，自动计算sim_min/sim_max
    if normalize:
        if sim_min is None or sim_max is None:
            # 计算所有案例的top1 sim_max分布
            pos_emb = pos_embeddings
            neg_emb = neg_embeddings
            all_emb = np.vstack([pos_emb, neg_emb])
            norm = np.linalg.norm(all_emb, axis=1, keepdims=True)
            norm_emb = all_emb / norm
            sim_matrix = norm_emb @ norm_emb.T
            # 计算每个案例的top1相似度
            sim_max_list = []
            for i in range(sim_matrix.shape[0]):
                row = sim_matrix[i].copy()
                row[i] = 0
                sim_max_list.append(row.max())
            auto_sim_min = np.min(sim_max_list)
            auto_sim_max = np.max(sim_max_list)
            sim_min = sim_min if sim_min is not None else auto_sim_min
            sim_max = sim_max if sim_max is not None else auto_sim_max
            # 设置参数
            pos_retriever.set_normalization_params(sim_min, sim_max)
            neg_retriever.set_normalization_params(sim_min, sim_max)
            logger.info(f"Auto computed normalization params: sim_min={sim_min:.4f}, sim_max={sim_max:.4f}")
    
    adaptive_retriever = AdaptiveRAGRetriever(pos_retriever, neg_retriever)
    
    logger.info(f"Loaded positive index: {len(pos_cases)} cases (min_k={min_k}, max_k={max_k}, alpha={alpha}, normalize={normalize})")
    logger.info(f"Loaded negative index: {len(neg_cases)} cases (min_k={min_k}, max_k={max_k}, alpha={alpha}, normalize={normalize})")
    
    return adaptive_retriever, embedding_model


def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs RAG on same test set')
    parser.add_argument('--test-file', type=str, default='data/final_all_data/first_stage/test.json',
                       help='CAIL2018 test.json path')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--k-positive', type=int, default=None,
                       help='Number of positive examples for RAG (None=adaptive)')
    parser.add_argument('--k-negative', type=int, default=None,
                       help='Number of negative examples for RAG (None=adaptive)')
    parser.add_argument('--min-k', type=int, default=1,
                       help='Minimum k for adaptive retrieval')
    parser.add_argument('--max-k', type=int, default=5,
                       help='Maximum k for adaptive retrieval')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Scaling factor for adaptive k: larger alpha = larger k')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize sim_max to get more uniform k distribution')
    parser.add_argument('--sim-min', type=float, default=None,
                       help='Minimum sim_max for normalization (auto detect if None)')
    parser.add_argument('--sim-max', type=float, default=None,
                       help='Maximum sim_max for normalization (auto detect if None)')
    parser.add_argument('--index-dir', type=str, default='data',
                       help='Index directory for RAG')
    parser.add_argument('--embedding-model', type=str, default='uer/sbert-base-chinese-nli',
                       help='Embedding model name')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 加载API配置
    base_url, api_key, model_name = load_api_config()
    
    # 加载标签映射
    charge_names, article_names = load_label_mappings()
    
    # 加载并采样测试数据（固定种子保证两次一样）
    test_data_full = DataLoader.load_cail2018(args.test_file)
    if args.max_samples < len(test_data_full):
        test_data = random.sample(test_data_full, args.max_samples)
    else:
        test_data = test_data_full
    logger.info(f"Sampled {len(test_data)} test samples (seed={args.seed})")
    
    # ========== 评估 baseline ==========
    print("\n" + "="*60)
    print("=== Evaluating Zero-Shot Baseline ===")
    print("="*60)
    baseline_model = ZeroShotLJPBaseline(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name
    )
    
    # 惰性加载标签映射（和evaluate_baseline保持一致）
    if not hasattr(baseline_model, 'charge_names'):
        charge_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/accu.txt')
        if os.path.exists(charge_path):
            with open(charge_path, 'r', encoding='utf-8') as f:
                baseline_model.charge_names = [line.strip() for line in f if line.strip()]
    
    if not hasattr(baseline_model, 'article_names'):
        article_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/law.txt')
        if os.path.exists(article_path):
            with open(article_path, 'r', encoding='utf-8') as f:
                baseline_model.article_names = [line.strip() for line in f if line.strip()]
    
    acc_charge_baseline, acc_article_baseline, p_baseline, r_baseline, f1_baseline, details_baseline = evaluate_model(
        baseline_model, test_data, charge_names, article_names
    )
    
    # ========== 评估 RAG ==========
    print("\n" + "="*60)
    if args.k_positive is None and args.k_negative is None:
        print(f"=== Evaluating RAG + Adaptive k (min={args.min_k}, max={args.max_k}, alpha={args.alpha}, normalize={args.normalize}) ===")
    else:
        print("=== Evaluating RAG + Fixed k (Positive/Negative) ===")
    print("="*60)
    
    # 如果开启normalize，自动计算sim_min/sim_max
    sim_min = args.sim_min
    sim_max = args.sim_max
    if args.normalize and (sim_min is None or sim_max is None):
        # 预计算所有案例的top1 sim_max，得到范围
        import numpy as np
        pos_emb = np.load(os.path.join(args.index_dir, "pos_index.npy"), allow_pickle=False)
        neg_emb = np.load(os.path.join(args.index_dir, "neg_index.npy"), allow_pickle=False)
        all_emb = np.vstack([pos_emb, neg_emb])
        norm = np.linalg.norm(all_emb, axis=1, keepdims=True)
        norm_emb = all_emb / norm
        sim_matrix = norm_emb @ norm_emb.T
        # 计算每个案例的top1 sim_max
        sim_max_list = []
        for i in range(sim_matrix.shape[0]):
            row = sim_matrix[i].copy()
            row[i] = 0
            sim_max_list.append(row.max())
        sim_min = np.min(sim_max_list)
        sim_max = np.max(sim_max_list)
        logger.info(f"Auto computed sim_min={sim_min:.4f}, sim_max={sim_max:.4f}")
    
    adaptive_retriever, embedding_model = load_rag_index(
        args.index_dir, args.embedding_model, args.min_k, args.max_k, args.alpha, args.adaptive_mode, sim_min, sim_max
    )
    
    rag_model = LJPAgentWithRAG(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        adaptive_retriever=adaptive_retriever,
        k_positive=args.k_positive,
        k_negative=args.k_negative,
        charge_names=charge_names,
        article_names=article_names
    )
    
    acc_charge_rag, acc_article_rag, p_rag, r_rag, f1_rag, details_rag = evaluate_model(
        rag_model, test_data, charge_names, article_names, embedding_model=embedding_model
    )
    
    # ========== 对比结果 ==========
    print("\n" + "="*60)
    print("=== Comparison Result (Same {} samples) ===".format(len(test_data)))
    print("="*60)
    print(f"{'Metric':<20} {'Baseline':<12} {'RAG':<12} {'Delta':<10}")
    print("-"*60)
    print(f"{'Charge Accuracy':<20} {acc_charge_baseline:<12.4f} {acc_charge_rag:<12.4f} {acc_charge_rag - acc_charge_baseline:>+10.4f}")
    print(f"{'Article Accuracy':<20} {acc_article_baseline:<12.4f} {acc_article_rag:<12.4f} {acc_article_rag - acc_article_baseline:>+10.4f}")
    print(f"{'Micro Precision':<20} {p_baseline:<12.4f} {p_rag:<12.4f} {p_rag - p_baseline:>+10.4f}")
    print(f"{'Micro Recall':<20} {r_baseline:<12.4f} {r_rag:<12.4f} {r_rag - r_baseline:>+10.4f}")
    print(f"{'Micro F1':<20} {f1_baseline:<12.4f} {f1_rag:<12.4f} {f1_rag - f1_baseline:>+10.4f}")
    print("="*60)
    
    # 保存详细对比
    output = {
        "settings": {
            "max_samples": args.max_samples,
            "seed": args.seed,
            "k_positive": args.k_positive,
            "k_negative": args.k_negative,
        },
        "baseline": {
            "acc_charge": acc_charge_baseline,
            "acc_article": acc_article_baseline,
            "precision": p_baseline,
            "recall": r_baseline,
            "f1": f1_baseline,
            "details": details_baseline
        },
        "rag": {
            "acc_charge": acc_charge_rag,
            "acc_article": acc_article_rag,
            "precision": p_rag,
            "recall": r_rag,
            "f1": f1_rag,
            "details": details_rag
        },
        "comparison": {
            "delta_acc_charge": acc_charge_rag - acc_charge_baseline,
            "delta_acc_article": acc_article_rag - acc_article_baseline,
            "delta_f1": f1_rag - f1_baseline
        }
    }
    
    # 保存结果到results文件夹，文件名包含配置参数方便对比
    os.makedirs("results", exist_ok=True)
    # 区分固定k和自适应k, 方便消融对比
    if args.k_positive is None and args.k_negative is None:
        # 全自适应
        output_file = os.path.join(
            "results", 
            f"comparison_s{args.seed}_n{args.max_samples}_adaptive_min{args.min_k}_max{args.max_k}.json"
        )
    elif args.k_positive is not None and args.k_negative is not None:
        # 全固定k
        output_file = os.path.join(
            "results", 
            f"comparison_s{args.seed}_n{args.max_samples}_p{args.k_positive}_neg{args.k_negative}_fixed.json"
        )
    else:
        # 混合：部分固定部分自适应
        p_tag = f"p{args.k_positive}" if args.k_positive is not None else "padaptive"
        n_tag = f"neg{args.k_negative}" if args.k_negative is not None else "negadaptive"
        output_file = os.path.join(
            "results", 
            f"comparison_s{args.seed}_n{args.max_samples}_{p_tag}_{n_tag}_min{args.min_k}_max{args.max_k}.json"
        )
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Full comparison saved to {output_file}")
    print(f"\nFull detailed comparison saved to: {output_file}")


if __name__ == "__main__":
    main()
