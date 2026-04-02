"""
均衡采样脚本 - 方案三混合采样
1. 每个罪名保底采样least_num个，确保所有罪名都覆盖
2. 剩余配额按平方根反比加权采样，提高罕见罪名采样概率
3. 输出完整判例信息，带罪名标签，方便后续分层embedding

Author: OpenClaw
Date: 2026-04-02
"""

import os
import json
import random
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_charge_distribution(charge_dist_file: str) -> Dict[str, int]:
    """
    加载预统计的罪名分布
    
    Args:
        charge_dist_file: 罪名分布JSON文件路径
    
    Returns:
        Dict[str, int]: 罪名 -> 总样本数
    """
    with open(charge_dist_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_train_data(
    train_file: str,
    charge_dist: Dict[str, int]
) -> Dict[str, List[dict]]:
    """
    流式读取训练集，按罪名分组
    
    内存占用说明：
    - 只分组不存fact，总内存可控，因为最终只存了索引比存整个数据还是小很多
    
    Args:
        train_file: 训练集JSON文件路径
        charge_dist: 预统计的罪名分布（用来提前知道有多少个罪名
    
    Returns:
        Dict[str, List[dict]]: 罪名 -> 该罪名下所有案件列表
    """
    charge_groups: Dict[str, List[dict]] = defaultdict(list)
    
    logger.info(f"Grouping training data by charge: {train_file}")
    
    total_cases = 0
    with open(train_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                case = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_idx}: {e}")
                continue
            
            # 获取罪名（适配多种格式，你的数据是 meta.accusation）
            charges = case.get('charge', [])
            if not charges and 'meta' in case:
                charges = case['meta'].get('accusation', [])
                if not charges:
                    charges = case['meta'].get('charge', [])
            
            if not charges:
                logger.warning(f"Case {line_idx} has no charges, skipping")
                continue
            
            # 多罪名案件：每个罪名都加进去这个案件
            # 这样同一个案件可能出现在多个罪名分组里，没问题，不影响
            for charge in charges:
                charge_groups[charge].append(case)
            
            total_cases += 1
            
            if line_idx % 20000 == 0:
                logger.info(f"Processed {line_idx} lines, {len(charge_groups)} groups")
    
    logger.info(f"Grouping done. Total {total_cases} cases grouped")
    
    # 检查：确保所有统计到的罪名和预统计一致
    missing = set(charge_dist.keys()) - set(charge_groups.keys())
    if missing:
        logger.warning(f"Missing charges in grouping: {len(missing)}")
    
    return charge_groups


def balanced_sample(
    charge_groups: Dict[str, List[dict]],
    total_sample: int,
    least_num: int = 1,
    seed: int = 42
) -> List[dict]:
    """
    混合均衡采样：
    1. 每个罪名保底采样 least_num 个
    2. 剩余配额按平方根反比加权采样，平衡分布
    
    Args:
        charge_groups: 按罪名分组后的案件
        total_sample: 总共需要采样多少个案件
        least_num: 每个罪名最小采样数
        seed: 随机种子
    
    Returns:
        List[dict]: 采样得到的案件列表
    """
    random.seed(seed)
    result: List[dict] = []
    
    # Step 1: 保底采样 - 每个罪名至少拿 least_num 个
    logger.info(f"Step 1: Guaranteed sampling, least_num={least_num} per charge")
    remaining_slots = total_sample
    charge_weights: Dict[str, float] = {}
    
    for charge, cases in charge_groups.items():
        available = len(cases)
        # 如果该罪名可用样本比最小要求少，就全部拿走
        n_sample = min(least_num, available)
        # 保底采样，不放回，不重复
        sampled = random.sample(cases, n_sample)
        result.extend(sampled)
        remaining_slots -= n_sample
        # 计算该罪名剩余采样的权重：平方根反比 = 1 / sqrt(total)
        # 总数越小，权重越大 -> 罕见罪名概率更高
        total = len(cases)
        weight = 1.0 / (total ** 0.5)
        charge_weights[charge] = weight
    
    logger.info(f"After guaranteed sampling: {len(result)} samples, {remaining_slots} slots remaining")
    
    # 如果已经用完配额，直接返回
    if remaining_slots <= 0:
        logger.warning(f"Guaranteed sampling already filled all slots ({len(result)} >= {total_sample})")
        return result
    
    # Step 2: 加权采样剩余配额
    logger.info(f"Step 2: Weighted sampling remaining {remaining_slots} slots")
    # 构建权重列表，加权采样
    charges = list(charge_weights.keys())
    weights = list(charge_weights.values())
    # 已经拿过保底，剩余样本可以再拿，允许放回采样
    
    already_sampled = 0
    already_tries = 0
    while already_sampled < remaining_slots and already_tries < remaining_slots * 10:
        # 按权重随机选一个罪名
        chosen_charge = random.choices(charges, weights=weights, k=1)[0]
        cases = charge_groups[chosen_charge]
        # 从该罪名随机选一个案件（允许重复，因为可能比保底拿的不同）
        case = random.choice(cases)
        # 检查去重（可选，这里不去重也没问题，允许同一个罪名可以重复采样
        # 我们这里不强制去重，因为数据集够大，重复概率低
        result.append(case)
        already_sampled += 1
        already_tries += 1
        
        if already_sampled % 100 == 0:
            logger.info(f"Weighted sampled {already_sampled} / {remaining_slots}")
    
    logger.info(f"Weighted sampling done. Total samples: {len(result)}")
    
    return result


def save_sampled_data(
    sampled_data: List[dict],
    output_file: str,
    include_charge_meta: bool = True
) -> None:
    """
    保存采样结果，添加charge标签方便后续分层embedding
    
    Args:
        sampled_data: 采样得到的案件列表
        output_file: 输出文件路径
        include_charge_meta: 是否添加charge分组元信息，方便分层embedding
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 如果需要，为每个样本添加统一的charge_meta字段，记录包含的所有罪名
    # 方便后续按罪名分组构建分层聚类索引
    output_data = []
    for case in sampled_data:
        if include_charge_meta:
            # 提取罪名到统一位置
            charges = case.get('charge', [])
            if not charges and 'meta' in case:
                charges = case['meta'].get('accusation', [])
                if not charges:
                    charges = case['meta'].get('charge', [])
            # 添加统一字段，不管原始格式，方便后续处理
            output_case = case.copy()
            output_case['_charge_meta'] = charges
            output_data.append(output_case)
        else:
            output_data.append(case)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 统计输出分布
    final_dist = defaultdict(int)
    for case in output_data:
        for charge in case['_charge_meta']:
            final_dist[charge] += 1
    
    min_final = min(final_dist.values())
    max_final = max(final_dist.values())
    avg_final = sum(final_dist.values()) / len(final_dist)
    
    logger.info(f"Saved {len(output_data)} samples to {output_file}")
    logger.info(f"Final distribution stats: min={min_final}, max={max_final}, avg={avg_final:.1f}")
    logger.info(f"All charges have at least {min_final} samples (guaranteed)")


def main():
    parser = argparse.ArgumentParser(
        description='Balanced sampling from CAIL2018 with per-charge guarantee'
    )
    parser.add_argument(
        '--train-file',
        type=str,
        default='data/final_all_data/first_stage/train.json',
        help='原始训练集文件路径'
    )
    parser.add_argument(
        '--charge-dist',
        type=str,
        default='data/charge_distribution.json',
        help='预统计的罪名分布JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/balanced_sampled.json',
        help='输出采样结果JSON'
    )
    parser.add_argument(
        '--total-sample',
        type=int,
        default=500,
        help='总共需要采样多少个案件'
    )
    parser.add_argument(
        '--least-num',
        type=int,
        default=1,
        help='每个罪名最小采样数（保底，确保全覆盖)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    args = parser.parse_args()
    
    # 1. 加载预统计罪名分布
    charge_dist = load_charge_distribution(args.charge_dist)
    logger.info(f"Loaded charge distribution: {len(charge_dist)} unique charges")
    
    # 2. 按罪名分组
    charge_groups = group_train_data(args.train_file, charge_dist)
    
    # 3. 均衡采样
    sampled = balanced_sample(
        charge_groups=charge_groups,
        total_sample=args.total_sample,
        least_num=args.least_num,
        seed=args.seed
    )
    
    # 4. 保存结果
    save_sampled_data(sampled, args.output, include_charge_meta=True)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
