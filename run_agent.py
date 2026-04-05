"""
单独运行LJP RAG Agent (no baseline)
支持批量测试和单案件预测

配置从config.json读取，命令行参数可以覆盖配置
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

from agent import Case, DataLoader, LJPAgentWithRAG, PredictionResult
from retriever import (
    FlatAdaptiveRAGRetriever,
    HierarchicalAdaptiveRAGRetriever,
    create_retriever_from_config,
)

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
            
            # 记录自适应信息和选中的案例（方便检查）
            if hasattr(result, 'retrieval_info'):
                detail["adaptive_k"] = {
                    "positive_k": result.retrieval_info.positive_k,
                    "negative_k": result.retrieval_info.negative_k,
                    "max_sim_pos": float(result.retrieval_info.max_sim_pos),
                    "max_sim_neg": float(result.retrieval_info.max_sim_neg),
                }
                # 保存选中的正负案例完整信息
                detail["retrieved_examples"] = {
                    "positive": [
                        {
                            "fact": case.fact,
                            "charges": case.charges,
                            "articles": case.articles,
                            "is_positive": case.is_positive,
                        }
                        for case in result.retrieval_info.positive_examples
                    ],
                    "negative": [
                        {
                            "fact": case.fact,
                            "charges": case.charges,
                            "articles": case.articles,
                            "is_positive": case.is_positive,
                        }
                        for case in result.retrieval_info.negative_examples
                    ],
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
    config: dict,
    llm_client=None,
    llm_model=None,
) -> tuple[FlatAdaptiveRAGRetriever | HierarchicalAdaptiveRAGRetriever, any]:
    """从配置加载RAG索引
    支持两种模式：
    - flat: 平面索引，所有案例放一起
    - hierarchical: 分层索引，按罪名分组，先筛罪名再找相似
    
    Args:
        config: 全局配置字典
        llm_client: 大模型客户端，llm模式需要
        llm_model: 大模型名称，llm模式需要
    
    Returns:
        (adaptive_retriever, embedding_model)
    """
    index_config = config.get("index", {})
    index_dir = index_config.get("index_dir", "data")
    embedding_model_name = index_config.get("embedding_model", "uer/sbert-base-chinese-nli")
    
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(embedding_model_name)
    logger.info(f"[初始化] 加载Embedding模型完成: {embedding_model_name}")
    
    # 从配置读取检索模式，create_retriever_from_config自动创建对应类型
    retriever_config = config.get("retriever", {})
    retriever_mode = retriever_config.get("mode", "flat")
    adaptive_retriever = create_retriever_from_config(
        retriever_config,
        llm_client=llm_client,
        llm_model=llm_model,
    )
    
    if retriever_mode == "flat":
        # ========== 平面索引模式 ==========
        logger.info("[初始化] 使用平面检索模式 (flat)")
        # 数据加载已经在create_retriever_from_config内部完成，使用index_clustered全量数据
        # pos: 2000 cases, neg: 2000 cases from data/index_clustered
        
        return adaptive_retriever, embedding_model
    
    elif retriever_mode == "hierarchical":
        # ========== 分层索引模式 ==========
        logger.info("[初始化] 使用分层检索模式 (hierarchical)")
        # 分层索引已经在创建时自动加载完毕了（create_retriever_from_config内部处理）
        
        # 统计计数打日志
        pos_count = sum(len(v) for v in adaptive_retriever.pos_retriever.cases_map.values())
        neg_count = sum(len(v) for v in adaptive_retriever.neg_retriever.cases_map.values())
        logger.info(f"[初始化][聚类分层模式] 加载完成: 正例={pos_count} ({len(adaptive_retriever.pos_retriever.cluster_list)} 个簇), 负例={neg_count} ({len(adaptive_retriever.neg_retriever.cluster_list)} 个簇)")
        
        return adaptive_retriever, embedding_model
    
    else:
        raise ValueError(f"Unknown retriever mode: {retriever_mode}, expected 'flat' or 'hierarchical'")


def load_api_config(config: dict):
    """加载API配置
    - base_url: 直接填写base url
    - api_key_env: 从环境变量读取api key（保护隐私，不放在配置里）
    - model_name: 直接填写模型名称
    """
    api_config = config.get("api", {})
    # base_url 和 model_name 直接从配置读，api_key 从环境变量读
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
    parser = argparse.ArgumentParser(description='Run LJP RAG Agent (no baseline), config loaded from config.json')
    # 配置文件
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径，默认=config.json')
    # 批量测试参数
    parser.add_argument('--test-file', type=str, default=None, help='CAIL2018 test.json path，覆盖配置文件')
    parser.add_argument('--max-samples', type=int, default=None, help='Number of samples to evaluate (None=run single, not batch)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for sampling，覆盖配置文件')
    # 单案件参数
    parser.add_argument('--input', type=str, help='输入案件文件 (json格式，包含fact字段)')
    parser.add_argument('--fact', type=str, help='直接输入案件事实文本')
    # 固定k覆盖（不使用自适应）
    parser.add_argument('--k-positive', type=int, default=None, help='固定正例数，None=使用配置自适应')
    parser.add_argument('--k-negative', type=int, default=None, help='固定负例数，None=使用配置自适应')
    # 索引
    parser.add_argument('--index-dir', type=str, default=None, help='索引目录，覆盖配置文件')
    parser.add_argument('--embedding-model', type=str, default=None, help='embedding模型名称，覆盖配置文件')
    # 输出
    parser.add_argument('--output', type=str, default=None, help='输出结果到json文件')
    # 检索模式（覆盖配置文件）
    parser.add_argument('--mode', type=str, default=None, help='检索模式: flat / hierarchical，覆盖配置文件')
    args = parser.parse_args()
    
    # 读取配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置日志级别
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    
    # 加载API配置
    base_url, api_key, model_name = load_api_config(config)
    
    # 覆盖索引配置
    if args.index_dir is not None:
        config.setdefault("index", {})["index_dir"] = args.index_dir
    if args.embedding_model is not None:
        config.setdefault("index", {})["embedding_model"] = args.embedding_model
    # 覆盖检索模式
    if args.mode is not None:
        config.setdefault("retriever", {})["mode"] = args.mode
    
    # 加载RAG索引（如果是llm模式，需要传入llm_client）
    from openai import OpenAI
    # 排除代理
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    llm_client = OpenAI(base_url=base_url, api_key=api_key)
    
    adaptive_retriever, embedding_model = load_rag_index(
        config,
        llm_client=llm_client,
        llm_model=model_name,
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
    else:
        article_names = []
    
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
        eval_config = config.get("evaluation", {})
        test_file = args.test_file or eval_config.get("test_file", 'data/final_all_data/first_stage/test.json')
        seed = args.seed or eval_config.get("seed", 42)
        
        print("\n" + "="*60)
        if args.k_positive is None and args.k_negative is None:
            print(f"=== Batch Evaluation: Adaptive k (LLM-verified) ===")
        else:
            print(f"=== Batch Evaluation: Fixed k (pos={args.k_positive}, neg={args.k_negative}) ===")
        print("="*60)
        # 加载并采样测试数据
        random.seed(seed)
        test_data_full = DataLoader.load_cail2018(test_file)
        if args.max_samples is not None and args.max_samples < len(test_data_full):
            test_data = random.sample(test_data_full, args.max_samples)
        else:
            test_data = test_data_full
        logger.info(f"Sampled {len(test_data)} test samples (seed={seed})")
        
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
        # 文件名：自动区分检索模式和k策略
        retriever_mode = config.get("retriever", {}).get("mode", "flat")
        if args.output is None:
            if args.k_positive is None and args.k_negative is None:
                output_file = os.path.join(
                    "results", 
                    f"{retriever_mode}_agent_s{seed}_n{args.max_samples}_llm-adaptive.json"
                )
            else:
                output_file = os.path.join(
                    "results", 
                    f"{retriever_mode}_agent_s{seed}_n{args.max_samples}_fixed_pos{args.k_positive}_neg{args.k_negative}.json"
                )
        else:
            output_file = args.output
        output = {
            "settings": {
                "config_file": args.config,
                "max_samples": args.max_samples,
                "k_positive": args.k_positive,
                "k_negative": args.k_negative,
                "seed": seed,
                "test_file": test_file,
                "retriever_config": config.get("retriever", {}),
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
            
            print(f"\nPredicted Charges: {', '.join(result.predicted_charges)}")
            print(f"Predicted Articles: {', '.join(result.predicted_articles)}")
            
            print("\nPredicted Judgment:")
            print(result.predicted_judgment)
            print()
            
            if hasattr(result, 'retrieval_info'):
                print("-"*60)
                print(f"Adaptive Retrieval Info:")
                print(f"  Positive k = {result.retrieval_info.positive_k}  |  Max similarity = {result.retrieval_info.max_sim_pos:.4f}")
                print(f"  Negative k = {result.retrieval_info.negative_k}  |  Max similarity = {result.retrieval_info.max_sim_neg:.4f}")
                print()
            
                # 打印检索到的案例摘要，方便快速检查
                print("Retrieved Positive Examples:")
                for i, case in enumerate(result.retrieval_info.positive_examples, 1):
                    fact_short = case.fact[:100] + "..." if len(case.fact) > 100 else case.fact
                    print(f"  [{i}] Charges: {', '.join(case.charges)} | Fact: {fact_short}")
                print()
                if result.retrieval_info.negative_examples:
                    print("Retrieved Negative Examples:")
                    for i, case in enumerate(result.retrieval_info.negative_examples, 1):
                        fact_short = case.fact[:100] + "..." if len(case.fact) > 100 else case.fact
                        print(f"  [{i}] Charges: {', '.join(case.charges)} | Fact: {fact_short}")
                    print()
            
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
                # 保存检索到的完整案例信息方便检查
                if hasattr(result, 'retrieval_info'):
                    output["retrieved_examples"] = {
                        "positive": [
                            {
                                "fact": case.fact,
                                "charges": case.charges,
                                "articles": case.articles,
                                "is_positive": case.is_positive,
                            }
                            for case in result.retrieval_info.positive_examples
                        ],
                        "negative": [
                            {
                                "fact": case.fact,
                                "charges": case.charges,
                                "articles": case.articles,
                                "is_positive": case.is_positive,
                            }
                            for case in result.retrieval_info.negative_examples
                        ],
                    }
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                logger.info(f"Result saved to {args.output}")
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
