"""
评估正负案例RAG LJPAgent在CAIL2018测试集上的性能
计算指标：
- 完全匹配准确率（Acc@charge, Acc@article）
- 精确率P、召回率R、F1-score
- 输出每个样本详细对比
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer

from agent import Case, DataLoader, LJPAgentWithRAG, PredictionResult
from retriever import EmbeddingRetriever
from main import load_api_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_charge(charge: str) -> str:
    """清洗罪名：去掉末尾的"罪"字"""
    charge = charge.strip()
    if charge.endswith("罪"):
        charge = charge[:-1]
    return charge


def clean_article(article: str) -> str:
    """清洗法条：提取编号，去掉多余文字"""
    article = article.strip()
    # 去掉所有非数字字符，只保留数字
    digits = ''.join([c for c in article if c.isdigit()])
    if digits:
        return digits
    # 如果没提取到数字，返回原字符串去掉前后缀
    article = article.replace("《中华人民共和国刑法》", "").replace("第", "").replace("条", "").replace("款", "").replace("第一款", "").strip()
    digits = ''.join([c for c in article if c.isdigit()])
    return digits if digits else article


def evaluate(
    model: LJPAgentWithRAG,
    test_data: List[dict],
    charge_names: List[str],
    article_names: List[str],
    embedding_model,
    max_samples: int = None
) -> Tuple[float, float, float, float, float, List[dict]]:
    """评估模型
    返回: (acc_charge, acc_article, precision, recall, f1, details)
    """
    correct_charges = 0
    correct_articles = 0
    total = 0
    details = []  # 保存每个样本的详细对比
    
    # TP/FP/FN 统计（微平均）
    tp = 0
    fp = 0
    fn = 0
    
    # 限制采样数量，方便快速测试
    if max_samples is not None and max_samples < len(test_data):
        import random
        random.seed(42)
        test_data = random.sample(test_data, max_samples)
    
    for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
        case = DataLoader.convert_to_case(item, is_positive=False)
        
        # 获取真实标签（已经是名称，不用再映射）
        true_charges_set = set(map(clean_charge, case.charges))
        true_articles_set = set(map(clean_article, case.articles))
        
        true_charges_list = list(true_charges_set)
        true_articles_list = list(true_charges_set)
        
        detail = {
            "index": i,
            "fact": case.fact,
            "true_charges": true_charges_list,
            "true_articles": true_articles_list,
            "predicted_charges": [],
            "predicted_articles": [],
            "predicted_judgment": "",
            "correct_charge": False,
            "correct_article": False,
            "error": None
        }
        
        try:
            result = model.predict(case, embedding_model)
            
            # 清洗预测结果
            pred_charges = set(map(clean_charge, result.predicted_charges))
            pred_articles = set(map(clean_article, result.predicted_articles))
            
            # 保存预测结果
            detail["predicted_charges"] = list(pred_charges)
            detail["predicted_articles"] = list(pred_articles)
            detail["predicted_judgment"] = result.predicted_judgment
            
            # 统计TP/FP/FN
            for c in pred_charges:
                if c in true_charges_set:
                    tp += 1
                else:
                    fp += 1
            for c in true_charges_set:
                if c not in pred_charges:
                    fn += 1
            
            # 判断是否完全正确（所有罪名/法条都预测对才算对）
            charge_correct = pred_charges == true_charges_set
            article_correct = pred_articles == true_articles_set
            
            if charge_correct:
                correct_charges += 1
            if article_correct:
                correct_articles += 1
            
            detail["correct_charge"] = charge_correct
            detail["correct_article"] = article_correct
            
            total += 1
            
            # 每10个例子打印一次进度
            if (i + 1) % 10 == 0:
                acc_c = correct_charges / total if total > 0 else 0
                acc_a = correct_articles / total if total > 0 else 0
                if tp + fp > 0:
                    p = tp / (tp + fp)
                    r = tp / (tp + fn)
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    logger.info(f"Progress: {i+1}/{len(test_data)}, "
                               f"Charge Acc: {acc_c:.4f}, Article Acc: {acc_a:.4f}, "
                               f"Micro-F1: {f1:.4f}")
        
        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            detail["error"] = str(e)
        
        details.append(detail)
    
    acc_charge = correct_charges / total if total > 0 else 0
    acc_article = correct_articles / total if total > 0 else 0
    
    # 计算微平均P/R/F1
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return acc_charge, acc_article, precision, recall, f1, details


def load_index(
    index_dir: str = "data",
    embedding_model_name: str = "uer/sbert-base-chinese-nli"
) -> Tuple[EmbeddingRetriever, EmbeddingRetriever, SentenceTransformer]:
    """加载预构建的正负索引"""
    # 加载embedding模型
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # 加载正例索引
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
    
    pos_retriever = EmbeddingRetriever()
    pos_retriever.index(pos_cases, pos_embeddings)
    
    # 加载负例索引
    neg_cases_path = os.path.join(index_dir, "neg_cases.json")
    neg_index_path = os.path.join(index_dir, "neg_index.npy")
    
    with open(neg_cases_path, 'r', encoding='utf-8') as f:
        neg_cases_data = json.load(f)
    
    neg_cases = []
    for item in neg_cases_data:
        # build_index保存负例时，已经把wrong_charges放到charges里了
        neg_cases.append(Case(
            fact=item['fact'],
            charges=item['charges'],
            articles=item['articles'],
            judgment='',
            is_positive=False
        ))
    
    neg_embeddings = np.load(neg_index_path)
    
    neg_retriever = EmbeddingRetriever()
    neg_retriever.index(neg_cases, neg_embeddings)
    
    logger.info(f"Loaded positive index: {len(pos_cases)} cases")
    logger.info(f"Loaded negative index: {len(neg_cases)} cases")
    
    return pos_retriever, neg_retriever, embedding_model


def load_label_mappings(data_dir: str = "data/cail2018/baseline"):
    """加载罪名和法条标签映射"""
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG LJP Agent on CAIL2018')
    parser.add_argument('--test-file', type=str, default='data/final_all_data/first_stage/test.json',
                       help='CAIL2018 test.json path')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Maximum number of samples to evaluate (for quick test)')
    parser.add_argument('--k-positive', type=int, default=1,
                       help='Number of positive examples to retrieve')
    parser.add_argument('--k-negative', type=int, default=1,
                       help='Number of negative examples to retrieve')
    parser.add_argument('--index-dir', type=str, default='data',
                       help='Directory with pre-built embedding indexes')
    parser.add_argument('--embedding-model', type=str, default='uer/sbert-base-chinese-nli',
                       help='Embedding model name')
    args = parser.parse_args()
    
    # 加载API配置
    base_url, api_key, model_name = load_api_config()
    
    # 加载标签映射
    charge_names, article_names = load_label_mappings()
    
    # 加载预构建索引
    pos_retriever, neg_retriever, embedding_model = load_index(
        args.index_dir, args.embedding_model
    )
    
    # 初始化模型
    model = LJPAgentWithRAG(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        pos_retriever=pos_retriever,
        neg_retriever=neg_retriever,
        k_positive=args.k_positive,
        k_negative=args.k_negative,
        charge_names=charge_names,
        article_names=article_names
    )
    
    # 加载测试数据
    test_data = DataLoader.load_cail2018(args.test_file)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # 开始评估
    logger.info(f"Starting evaluation, max_samples={args.max_samples}")
    acc_charge, acc_article, precision, recall, f1, details = evaluate(
        model, test_data, charge_names, article_names, embedding_model, args.max_samples
    )
    
    # 输出结果
    print("\n" + "="*60)
    print("Evaluation Result (RAG + Positive/Negative Examples):")
    print(f"Total evaluated: {min(args.max_samples, len(test_data))} samples")
    print(f"完全匹配 - 罪名准确率: {acc_charge:.4f} ({acc_charge*100:.2f}%)")
    print(f"完全匹配 - 法条准确率: {acc_article:.4f} ({acc_article*100:.2f}%)")
    print(f"微平均 - 精确率P: {precision:.4f}")
    print(f"微平均 - 召回率R: {recall:.4f}")
    print(f"微平均 - F1-score: {f1:.4f}")
    print("="*60)
    
    # 保存结果到results文件夹
    os.makedirs("results", exist_ok=True)
    output_file = os.path.join("results", f"result_rag_{args.k_positive}pos_{args.k_negative}neg_{args.max_samples}.json")
    details_file = os.path.join("results", f"details_rag_{args.k_positive}pos_{args.k_negative}neg_{args.max_samples}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "mode": "rag-positive-negative",
            "k_positive": args.k_positive,
            "k_negative": args.k_negative,
            "max_samples": args.max_samples,
            "acc_charge": acc_charge,
            "acc_article": acc_article,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }, f, indent=2, ensure_ascii=False)
    
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump({
            "details": details
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Aggregate result saved to {output_file}")
    logger.info(f"Detailed predictions saved to {details_file}")


if __name__ == "__main__":
    main()
