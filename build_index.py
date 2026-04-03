"""
构建正负案例向量索引，支持两种模式：
1. flat: 传统平面索引，所有案例放一起（兼容旧版本）
2. hierarchical: 分层索引，按罪名分组存储，检索时先筛罪名再找相似

分层索引构建流程：
- 所有案例保存在同一个文件里，用charge分组标记，逻辑分层而非物理分层
- 加载时一次性读入所有案例，检索第一步根据候选罪名过滤范围，再找相似
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
    负例：按**正确罪名**分组，因为我们要找"正确罪名相同，事实相似，判决错误"的案例
    所以分组用 true_charges，不是 wrong_charges
    """
    cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            # 负例保持wrong标签，但分组按正确罪名
            case = Case(
                fact=item['fact'],
                charges=item['wrong_charges'],
                articles=item['wrong_articles'],
                judgment='',
                is_positive=False
            )
            # 保存正确罪名，用来分组
            case.true_charges = item['true_charges']
            cases.append(case)
    logger.info(f"Loaded {len(cases)} negative examples")
    return cases


def group_cases_by_charge(cases: List[Case], is_negative: bool = False) -> Dict[str, List[Case]]:
    """将案例按罪名分组
    一个案例可能有多个罪名，会放进多个分组里（因为一个案例涉及多个罪名，都要能被搜到）
    
    Args:
        cases: 案例列表
        is_negative: 是否为负例，负例按正确罪名分组，不是错的罪名
    """
    charge_map: Dict[str, List[Case]] = {}
    for case in cases:
        # 负例：按正确罪名分组（我们要在正确罪名下找错误判决对比）
        # 正例：正常按罪名分组
        if is_negative and hasattr(case, 'true_charges'):
            charges = case.true_charges
        else:
            charges = case.charges
        
        for charge in charges:
            charge = charge.strip()
            # 格式化罪名，去掉"罪"后缀统一命名
            if charge.endswith("罪"):
                charge = charge[:-1]
            if charge not in charge_map:
                charge_map[charge] = []
            charge_map[charge].append(case)
    
    logger.info(f"Grouped {len(cases)} cases into {len(charge_map)} distinct charges"
              f"{' (negative, grouped by true charge)' if is_negative else ''}")
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
    """保存分层索引：统一保存在几个文件中，逻辑分组
    - {prefix}_cases.json: 所有案例列表，每个案例带charge标记
    - {prefix}_index.npy: 所有embeddings拼接在一起
    - {prefix}_charge_offsets.json: 每个罪名在大数组中的起始偏移和长度
    - {prefix}_charge_list.json: 所有罪名列表
    """
    os.makedirs(output_root, exist_ok=True)
    
    # 收集所有案例，记录每个罪名的偏移
    all_cases = []
    all_embeddings = []
    charge_offsets = {}
    current_offset = 0
    
    # 按罪名顺序拼接
    all_charges = list(charge_map.keys())
    all_charges.sort()
    
    for charge in all_charges:
        cases = charge_map[charge]
        embeddings = embeddings_map[charge]
        n = len(cases)
        
        # 记录偏移信息
        charge_offsets[charge] = {
            "start": current_offset,
            "count": n
        }
        
        # 添加到总列表
        all_cases.extend([{
            "fact": case.fact,
            "charges": case.charges,
            "articles": case.articles,
            "is_positive": case.is_positive,
            "charge_group": charge
        } for case in cases])
        
        all_embeddings.append(embeddings)
        current_offset += n
    
    # 拼接所有embeddings
    concatenated = np.concatenate(all_embeddings, axis=0)
    
    # 保存罪名列表
    with open(os.path.join(output_root, f"{prefix}_charge_list.json"), 'w', encoding='utf-8') as f:
        json.dump(all_charges, f, ensure_ascii=False, indent=2)
    
    # 保存偏移信息
    with open(os.path.join(output_root, f"{prefix}_charge_offsets.json"), 'w', encoding='utf-8') as f:
        json.dump(charge_offsets, f, ensure_ascii=False, indent=2)
    
    # 保存所有案例
    with open(os.path.join(output_root, f"{prefix}_cases.json"), 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)
    
    # 保存所有embeddings
    np.save(os.path.join(output_root, f"{prefix}_index.npy"), concatenated)
    
    total_cases = len(all_cases)
    logger.info(f"Saved hierarchical {prefix} index: {len(all_charges)} charges, {total_cases} total cases")
    logger.info(f"  Files: {prefix}_charge_list.json, {prefix}_charge_offsets.json, {prefix}_cases.json, {prefix}_index.npy")


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
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, default='hierarchical', choices=['flat', 'hierarchical'],
                       help='Index mode: flat=all in one, hierarchical=group by charge (logical grouping)')
    parser.add_argument('--pos-input', type=str, default='data/sampled_positives.json',
                       help='Input positive samples json')
    parser.add_argument('--neg-input', type=str, default='data/generated_negatives.json',
                       help='Input generated negative examples json')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for index files (overrides config)')
    parser.add_argument('--embedding-model', type=str, default=None,
                       help='Sentence embedding model name (overrides config)')
    args = parser.parse_args()
    
    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    index_config = config.get("index", {})
    output_dir = args.output_dir or index_config.get("index_dir", "data")
    embedding_model = args.embedding_model or index_config.get("embedding_model", "uer/sbert-base-chinese-nli")
    
    # 加载模型
    logger.info(f"Loading embedding model: {embedding_model} (from config)")
    model = SentenceTransformer(embedding_model)
    
    if args.mode == 'flat':
        # ========== 旧版平面索引 ==========
        # 处理正例
        pos_cases = load_positives(args.pos_input)
        pos_embeddings = encode_cases(pos_cases, model)
        save_flat_index(pos_cases, pos_embeddings, output_dir, "pos")
        
        # 处理负例
        neg_cases = load_negatives(args.neg_input)
        neg_embeddings = encode_cases(neg_cases, model)
        save_flat_index(neg_cases, neg_embeddings, output_dir, "neg")
        
        logger.info("All done! Flat index built successfully.")
        logger.info(f"Positive index: {output_dir}/pos_cases.json + pos_index.npy")
        logger.info(f"Negative index: {output_dir}/neg_cases.json + neg_index.npy")
    
    else:
        # ========== 分层索引（逻辑分组，统一存储） ==========
        # 处理正例
        pos_cases = load_positives(args.pos_input)
        pos_charge_map = group_cases_by_charge(pos_cases, is_negative=False)
        
        # 每个罪名单独编码
        pos_embeddings_map: Dict[str, np.ndarray] = {}
        for charge, cases in pos_charge_map.items():
            logger.info(f"Encoding positive cases for charge: {charge} ({len(cases)} cases)")
            pos_embeddings_map[charge] = encode_cases(cases, model)
        
        # 处理负例：负例按正确罪名分组
        neg_cases = load_negatives(args.neg_input)
        neg_charge_map = group_cases_by_charge(neg_cases, is_negative=True)
        
        # 每个罪名单独编码
        neg_embeddings_map: Dict[str, np.ndarray] = {}
        for charge, cases in neg_charge_map.items():
            logger.info(f"Encoding negative cases for charge: {charge} ({len(cases)} cases)")
            neg_embeddings_map[charge] = encode_cases(cases, model)
        
        # 保存索引（统一文件）
        hierarchical_root = os.path.join(output_dir, "index_by_charge")
        save_hierarchical_index(pos_charge_map, pos_embeddings_map, hierarchical_root, "pos")
        save_hierarchical_index(neg_charge_map, neg_embeddings_map, hierarchical_root, "neg")
        
        logger.info("All done! Hierarchical index (logical grouping) built successfully.")
        logger.info(f"Index root: {hierarchical_root}")
        logger.info(f"  ├─ pos_charge_list.json    : 所有正例罪名列表")
        logger.info(f"  ├─ pos_charge_offsets.json: 每个罪名偏移信息")
        logger.info(f"  ├─ pos_cases.json         : 所有正例案例")
        logger.info(f"  └─ pos_index.npy          : 所有正例embeddings")


if __name__ == "__main__":
    main()
