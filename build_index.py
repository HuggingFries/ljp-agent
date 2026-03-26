"""
构建正负案例向量索引，支持两种模式：
1. flat: 传统平面索引，所有案例放一起（兼容旧版本）
2. hierarchical: 分层索引，按罪名分组存储，检索时先筛罪名再找相似

分层索引构建流程：
- 正例索引：data/sampled_positives.json → data/index_by_charge/{charge_name}/{pos/neg}_*.json
- 每个罪名单独保存embedding和cases，检索时只加载命中罪名的案例
"""

import os
import json
import numpy as np
import argparse
import logging
from typing import List, Dict

# 使用国内镜像加速huggingface下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer

from agent import Case

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_positives(file_path: str) -> List[Case]:
    """加载采样好的正例"""
    cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            cases.append(Case(
                fact=item['fact'],
                charges=item['charges'],
                articles=item['articles'],
                judgment='',
                is_positive=True
            ))
    logger.info(f"Loaded {len(cases)} positive examples")
    return cases


def load_negatives(file_path: str) -> List[Case]:
    """加载生成好的负例
    负例：把wrong_charges/wrong_articles当作标签，is_positive=False
    分组按正确罪名分组，因为我们要找"正确罪名相同，事实相似，判决错误"的案例
    """
    cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            cases.append(Case(
                fact=item['fact'],
                charges=item['wrong_charges'],
                articles=item['wrong_articles'],
                judgment='',
                is_positive=False
            ))
    # 保存原始正确罪名到案例，方便分组
    for case, item in zip(cases, data):
        case.true_charges = item['true_charges']
    logger.info(f"Loaded {len(cases)} negative examples")
    return cases


def group_cases_by_charge(cases: List[Case]) -> Dict[str, List[Case]]:
    """将案例按罪名分组
    一个案例可能有多个罪名，会放进多个分组里（因为一个案例涉及多个罪名，都要能被搜到）
    """
    charge_map: Dict[str, List[Case]] = {}
    for case in cases:
        for charge in case.charges:
            charge = charge.strip()
            # 格式化罪名，去掉"罪"后缀统一命名
            if charge.endswith("罪"):
                charge = charge[:-1]
            if charge not in charge_map:
                charge_map[charge] = []
            charge_map[charge].append(case)
    
    logger.info(f"Grouped {len(cases)} cases into {len(charge_map)} distinct charges")
    return charge_map


def encode_cases(cases: List[Case], model: SentenceTransformer) -> np.ndarray:
    """对所有case的fact做embedding"""
    embeddings = []
    for i, case in enumerate(cases):
        embedding = model.encode(case.fact, normalize_embeddings=True)
        embeddings.append(embedding)
        if (i + 1) % 100 == 0:
            logger.info(f"Encoded {i+1}/{len(cases)}")
    return np.array(embeddings)


def save_hierarchical_index(
    charge_map: Dict[str, List[Case]],
    embeddings_map: Dict[str, np.ndarray],
    output_root: str,
    prefix: str,  # "pos" or "neg"
):
    """保存分层索引：每个罪名一个子目录，存cases和embeddings"""
    os.makedirs(output_root, exist_ok=True)
    
    # 保存罪名列表
    all_charges = list(charge_map.keys())
    all_charges.sort()
    with open(os.path.join(output_root, f"{prefix}_charge_list.json"), 'w', encoding='utf-8') as f:
        json.dump(all_charges, f, ensure_ascii=False, indent=2)
    
    # 每个罪名单独保存
    total_cases = 0
    for charge in all_charges:
        charge_dir = os.path.join(output_root, prefix, charge.replace('/', '_'))
        os.makedirs(charge_dir, exist_ok=True)
        
        cases = charge_map[charge]
        embeddings = embeddings_map[charge]
        
        # 保存cases
        case_list = []
        for case in cases:
            case_list.append({
                "fact": case.fact,
                "charges": case.charges,
                "articles": case.articles,
                "is_positive": case.is_positive
            })
        
        with open(os.path.join(charge_dir, "cases.json"), 'w', encoding='utf-8') as f:
            json.dump(case_list, f, indent=2, ensure_ascii=False)
        
        # 保存embeddings
        np.save(os.path.join(charge_dir, "index.npy"), embeddings)
        
        total_cases += len(cases)
        logger.debug(f"Saved {len(cases)} cases for charge '{charge}'")
    
    logger.info(f"Saved hierarchical {prefix} index: {len(charge_map)} charges, {total_cases} total cases")


def save_flat_index(cases: List[Case], embeddings: np.ndarray, output_dir: str, prefix: str):
    """保存平面索引（兼容旧版本）"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存cases
    case_list = []
    for case in cases:
        case_list.append({
            "fact": case.fact,
            "charges": case.charges,
            "articles": case.articles,
            "is_positive": case.is_positive
        })
    
    with open(os.path.join(output_dir, f"{prefix}_cases.json"), 'w', encoding='utf-8') as f:
        json.dump(case_list, f, indent=2, ensure_ascii=False)
    
    # 保存embeddings
    np.save(os.path.join(output_dir, f"{prefix}_index.npy"), embeddings)
    
    logger.info(f"Saved {len(cases)} {prefix} flat index to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Build embedding index, support flat/hierarchical')
    parser.add_argument('--mode', type=str, default='flat', choices=['flat', 'hierarchical'],
                       help='Index mode: flat=all in one, hierarchical=group by charge')
    parser.add_argument('--pos-input', type=str, default='data/sampled_positives.json',
                       help='Input positive samples json')
    parser.add_argument('--neg-input', type=str, default='data/generated_negatives.json',
                       help='Input generated negative examples json')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for index files')
    parser.add_argument('--embedding-model', type=str, default='uer/sbert-base-chinese-nli',
                       help='Sentence embedding model name (from huggingface)')
    args = parser.parse_args()
    
    # 加载模型
    logger.info(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)
    
    if args.mode == 'flat':
        # ========== 旧版平面索引 ==========
        # 处理正例
        pos_cases = load_positives(args.pos_input)
        pos_embeddings = encode_cases(pos_cases, model)
        save_flat_index(pos_cases, pos_embeddings, args.output_dir, "pos")
        
        # 处理负例
        neg_cases = load_negatives(args.neg_input)
        neg_embeddings = encode_cases(neg_cases, model)
        save_flat_index(neg_cases, neg_embeddings, args.output_dir, "neg")
        
        logger.info("All done! Flat index built successfully.")
        logger.info(f"Positive index: {args.output_dir}/pos_cases.json + pos_index.npy")
        logger.info(f"Negative index: {args.output_dir}/neg_cases.json + neg_index.npy")
    
    else:
        # ========== 分层索引 ==========
        # 处理正例
        pos_cases = load_positives(args.pos_input)
        pos_charge_map = group_cases_by_charge(pos_cases)
        
        # 每个罪名单独编码
        pos_embeddings_map: Dict[str, np.ndarray] = {}
        for charge, cases in pos_charge_map.items():
            logger.info(f"Encoding positive cases for charge: {charge} ({len(cases)} cases)")
            pos_embeddings_map[charge] = encode_cases(cases, model)
        
        # 处理负例
        neg_cases = load_negatives(args.neg_input)
        neg_charge_map = group_cases_by_charge(neg_cases)
        
        # 每个罪名单独编码
        neg_embeddings_map: Dict[str, np.ndarray] = {}
        for charge, cases in neg_charge_map.items():
            logger.info(f"Encoding negative cases for charge: {charge} ({len(cases)} cases)")
            neg_embeddings_map[charge] = encode_cases(cases, model)
        
        # 保存索引
        pos_output_root = os.path.join(args.output_dir, "index_by_charge", "pos")
        save_hierarchical_index(pos_charge_map, pos_embeddings_map, os.path.join(args.output_dir, "index_by_charge"), "pos")
        save_hierarchical_index(neg_charge_map, neg_embeddings_map, os.path.join(args.output_dir, "index_by_charge"), "neg")
        
        logger.info("All done! Hierarchical index built successfully.")
        logger.info(f"Index root: {os.path.join(args.output_dir, 'index_by_charge')}")
        logger.info(f"  ├─ pos/: {len(pos_charge_map)} charges")
        logger.info(f"  └─ neg/: {len(neg_charge_map)} charges")


if __name__ == "__main__":
    main()
