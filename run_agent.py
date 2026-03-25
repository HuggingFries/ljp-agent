"""
单独运行LJP RAG Agent (no baseline)
支持批量测试和单案件预测
"""

import argparse
import json
import logging
import random
import os
import sys

# 自动添加用户site-packages路径
if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
    user_local_base = os.path.expanduser("~/.local/lib")
    if os.path.exists(user_local_base):
        for version_dir in os.listdir(user_local_base):
            if version_dir.startswith("python"):
                user_local_lib = os.path.join(user_local_base, version_dir, "site-packages")
                if user_local_lib not in sys.path:
                    sys.path.insert(0, user_local_lib)
                break

try:
    from dotenv import load_dotenv
except ImportError:
    print("⚠️  找不到模块 dotenv，请先安装依赖：")
    print("   conda create -n ljp-agent python=3.11")
    print("   conda activate ljp-agent")
    print("   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agent import Case, DataLoader, LJPAgentWithRAG, PredictionResult
from retriever import EmbeddingRetriever, AdaptiveRAGRetriever
from main import load_api_config


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


def evaluate_batch(
    agent: LJPAgentWithRAG,
    test_data: list[dict],
    charge_names: list[str],
    article_names: list[str],
    embedding_model
) -> tuple[float, float, float, float, float, list[dict]]:
    """批量评估，记录结果和自适应信息"""
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
        
        logger.info(f"[{i+1}/{total_samples}] Processing sample...")
        
        try:
            result = agent.predict(case, embedding_model)
            pred_charges = set(map(clean_charge, result.predicted_charges))
            pred_articles = set(map(clean_article, result.predicted_articles))
            
            detail["predicted_charges"] = list(pred_charges)
            detail["predicted_articles"] = list(pred_articles)
            detail["predicted_judgment"] = result.predicted_judgment
            
            # 记录自适应信息
            if hasattr(result, 'retrieval_info'):
                detail["adaptive_k"] = {
                    "positive_k": result.retrieval_info.positive_k,
                    "negative_k": result.retrieval_info.negative_k,
                    "max_sim_pos": float(result.retrieval_info.max_sim_pos),
                    "max_sim_neg": float(result.retrieval_info.max_sim_neg),
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
        
        details.append(detail)
    
    acc_charge = correct_charges / total if total > 0 else 0
    acc_article = correct_articles / total if total > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return acc_charge, acc_article, precision, recall, f1, details


def load_rag_index(
    index_dir: str = "data",
    embedding_model_name: str = "uer/sbert-base-chinese-nli",
    min_k: int = 1,
    max_k: int = 5,
    alpha: float = 1.0,
    normalize: bool = False,
    sim_min: float | None = None,
    sim_max: float | None = None,
) -> tuple[AdaptiveRAGRetriever, any]:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from sentence_transformers import SentenceTransformer
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
    
    import numpy as np
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
    
    # 如果开启normalize，自动计算参数
    if normalize:
        if (sim_min is None or sim_max is None):
            import numpy as np
            pos_emb = pos_embeddings
            neg_emb = neg_embeddings
            all_emb = np.vstack([pos_emb, neg_emb])
            norm = np.linalg.norm(all_emb, axis=1, keepdims=True)
            norm_emb = all_emb / norm
            sim_matrix = norm_emb @ norm_emb.T
            sim_max_list = []
            for i in range(sim_matrix.shape[0]):
                row = sim_matrix[i].copy()
                row[i] = 0
                sim_max_list.append(row.max())
            auto_sim_min = np.min(sim_max_list)
            auto_sim_max = np.max(sim_max_list)
            sim_min = sim_min if sim_min is not None else auto_sim_min
            sim_max = sim_max if sim_max is not None else auto_sim_max
            pos_retriever.set_normalization_params(sim_min, sim_max)
            neg_retriever.set_normalization_params(sim_min, sim_max)
            logger.info(f"Auto computed normalization params: sim_min={sim_min:.4f}, sim_max={sim_max:.4f}")
    
    adaptive_retriever = AdaptiveRAGRetriever(pos_retriever, neg_retriever)
    
    logger.info(f"Loaded positive index: {len(pos_cases)} cases (min_k={min_k}, max_k={max_k}, alpha={alpha}, normalize={normalize})")
    logger.info(f"Loaded negative index: {len(neg_cases)} cases (min_k={min_k}, max_k={max_k}, alpha={alpha}, normalize={normalize})")
    
    return adaptive_retriever, embedding_model


def main():
    parser = argparse.ArgumentParser(description='Run LJP RAG Agent (no baseline)')
    # 批量测试参数（和compare一致）
    parser.add_argument('--test-file', type=str, default='data/final_all_data/first_stage/test.json',
                       help='CAIL2018 test.json path (default: data/final_all_data/first_stage/test.json)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Number of samples to evaluate (None=run single, not batch)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    # 单案件参数
    parser.add_argument('--input', type=str, help='输入案件文件 (json格式，包含fact字段)')
    parser.add_argument('--fact', type=str, help='直接输入案件事实文本')
    # 检索参数
    parser.add_argument('--k-positive', type=int, default=None, help='固定正例数，None=自适应')
    parser.add_argument('--k-negative', type=int, default=None, help='固定负例数，None=自适应')
    parser.add_argument('--min-k', type=int, default=1, help='自适应最小k')
    parser.add_argument('--max-k', type=int, default=5, help='自适应最大k')
    parser.add_argument('--alpha', type=float, default=1.0, help='Scaling factor for adaptive k: larger alpha = larger k')
    parser.add_argument('--normalize', action='store_true', help='Normalize sim_max to get more uniform k distribution')
    parser.add_argument('--sim-min', type=float, default=None, help='Minimum sim_max for normalization (auto detect if None)')
    parser.add_argument('--sim-max', type=float, default=None, help='Maximum sim_max for normalization (auto detect if None)')
    parser.add_argument('--index-dir', type=str, default='data', help='索引目录')
    parser.add_argument('--embedding-model', type=str, default='uer/sbert-base-chinese-nli', help='embedding模型名称')
    # 输出
    parser.add_argument('--output', type=str, default=None, help='输出结果到json文件')
    args = parser.parse_args()
    
    # 加载API配置
    base_url, api_key, model_name = load_api_config()
    
    # 如果开启normalize，自动计算sim_min/sim_max
    sim_min = args.sim_min
    sim_max = args.sim_max
    if args.normalize and (sim_min is None or sim_max is None):
        import numpy as np
        pos_emb = np.load(os.path.join(args.index_dir, "pos_index.npy"), allow_pickle=False)
        neg_emb = np.load(os.path.join(args.index_dir, "neg_index.npy"), allow_pickle=False)
        all_emb = np.vstack([pos_emb, neg_emb])
        norm = np.linalg.norm(all_emb, axis=1, keepdims=True)
        norm_emb = all_emb / norm
        sim_matrix = norm_emb @ norm_emb.T
        sim_max_list = []
        for i in range(sim_matrix.shape[0]):
            row = sim_matrix[i].copy()
            row[i] = 0
            sim_max_list.append(float(row.max()))
        auto_sim_min = float(np.min(sim_max_list))
        auto_sim_max = float(np.max(sim_max_list))
        sim_min = sim_min if sim_min is not None else auto_sim_min
        sim_max = sim_max if sim_max is not None else auto_sim_max
    
    # 加载RAG索引
    adaptive_retriever, embedding_model = load_rag_index(
        args.index_dir, args.embedding_model, args.min_k, args.max_k, args.alpha, args.normalize, sim_min, sim_max
    )
    
    # 加载标签映射
    charge_names = []
    charge_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/accu.txt')
    if os.path.exists(charge_path):
        with open(charge_path, 'r', encoding='utf-8') as f:
            charge_names = [line.strip() for line in f if line.strip()]
    
    article_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/law.txt')
    if os.path.exists(article_path):
        with open(article_path, 'r', encoding='utf-8') as f:
            article_names = [line.strip() for line in f if line.strip()]
    
    # 初始化agent
    agent = LJPAgentWithRAG(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        adaptive_retriever=adaptive_retriever,
        k_positive=args.k_positive,
        k_negative=args.k_negative,
        charge_names=charge_names,
        article_names=article_names
    )
    
    # ========== 批量测试模式 ==========
    if args.max_samples is not None:
        print("\n" + "="*60)
        if args.k_positive is None and args.k_negative is None:
            print(f"=== Batch Evaluation: Adaptive k (min={args.min_k}, max={args.max_k}, alpha={args.alpha}, normalize={args.normalize}) ===")
        else:
            print(f"=== Batch Evaluation: Fixed k (pos={args.k_positive}, neg={args.k_negative}) ===")
        print("="*60)
        # 加载并采样测试数据
        random.seed(args.seed)
        test_data_full = DataLoader.load_cail2018(args.test_file)
        if args.max_samples is not None and args.max_samples < len(test_data_full):
            test_data = random.sample(test_data_full, args.max_samples)
        else:
            test_data = test_data_full
        logger.info(f"Sampled {len(test_data)} test samples (seed={args.seed})")
        
        # 评估
        acc_charge, acc_article, precision, recall, f1, details = evaluate_batch(
            agent, test_data, charge_names, article_names, embedding_model
        )
        
        # 输出结果
        print("\n" + "="*60)
        print(f"=== Evaluation Result ({len(test_data)} samples) ===")
        print("="*60)
        print(f"{'Metric':<20} {'Score':<12}")
        print("-"*60)
        print(f"{'Charge Accuracy':<20} {acc_charge:<12.4f}")
        print(f"{'Article Accuracy':<20} {acc_article:<12.4f}")
        print(f"{'Micro Precision':<20} {precision:<12.4f}")
        print(f"{'Micro Recall':<20} {recall:<12.4f}")
        print(f"{'Micro F1':<20} {f1:<12.4f}")
        print("="*60)
        
        # 保存结果
        os.makedirs("results", exist_ok=True)
        # 文件名保持和compare一致
        if args.k_positive is None and args.k_negative is None:
            output_file = os.path.join(
                "results", 
                f"agent_s{args.seed}_n{args.max_samples}_adaptive_min{args.min_k}_max{args.max_k}.json"
            )
        elif args.output is None:
            output_file = os.path.join(
                "results", 
                f"agent_s{args.seed}_n{args.max_samples}_fixed_pos{args.k_positive}_neg{args.k_negative}.json"
            )
        else:
            output_file = args.output
        output = {
            "settings": {
                "max_samples": args.max_samples,
                "k_positive": args.k_positive,
                "k_negative": args.k_negative,
                "min_k": args.min_k,
                "max_k": args.max_k,
                "alpha": args.alpha,
                "normalize": args.normalize,
                "sim_min": float(sim_min) if sim_min is not None else None,
                "sim_max": float(sim_max) if sim_max is not None else None,
                "seed": args.seed,
                "test_file": args.test_file,
                "embedding_model": args.embedding_model,
            },
            "metrics": {
                "acc_charge": acc_charge,
                "acc_article": acc_article,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            "details": details,
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    # ========== 单案件预测模式 ==========
    else:
        if args.input is None and args.fact is None:
            print("\n" + "="*60)
            print("=== Single Case Prediction Mode ===")
            print("="*60)
            print("Please provide either --input <json_file> or --fact \"<case_fact_text>\"")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("=== Single Case Prediction ===")
        print("="*60)
        
        # 加载案件事实
        if args.input is not None:
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
                fact = data.get('fact', '')
        else:
            fact = args.fact
        
        if not fact:
            logger.error("No fact provided")
            sys.exit(1)
        
        print(f"\nCase Fact:\n{fact}\n")
        logger.info("Running prediction...")
        
        case = Case(
            fact=fact,
            charges=[],
            articles=[],
            judgment='',
            is_positive=False
        )
        
        try:
            result = agent.predict(case, embedding_model)
            
            print("\n" + "="*60)
            print("=== Prediction Result ===")
            print("="*60)
            print(f"Predicted Charges: {', '.join(result.predicted_charges)}")
            print(f"Predicted Articles: {', '.join(result.predicted_articles)}")
            print("\nPredicted Judgment:")
            print(result.predicted_judgment)
            
            if hasattr(result, 'retrieval_info'):
                print("\n" + "-"*60)
                print(f"Adaptive Retrieval Info:")
                print(f"  Positive k: {result.retrieval_info.positive_k} (max_sim={result.retrieval_info.max_sim_pos:.4f})")
                print(f"  Negative k: {result.retrieval_info.negative_k} (max_sim={result.retrieval_info.max_sim_neg:.4f})")
            
            print("="*60)
            
            # 保存结果到文件
            if args.output is not None:
                output = {
                    "fact": fact,
                    "predicted_charges": result.predicted_charges,
                    "predicted_articles": result.predicted_articles,
                    "predicted_judgment": result.predicted_judgment,
                    "retrieval_info": {
                        "positive_k": result.retrieval_info.positive_k,
                        "negative_k": result.retrieval_info.negative_k,
                        "max_sim_pos": float(result.retrieval_info.max_sim_pos),
                        "max_sim_neg": float(result.retrieval_info.max_sim_neg),
                    } if hasattr(result, 'retrieval_info') else None,
                }
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                logger.info(f"Result saved to {args.output}")
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
