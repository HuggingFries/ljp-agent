"""
CAIL2018 罪名分布统计脚本
统计训练集中每个罪名的样本数量，输出分布统计报告
支持大文件流式读取，内存占用低

Author: OpenClaw
Date: 2026-04-02
"""

import os
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stats_charge_distribution(
    train_file: str,
    output_json: str = "charge_distribution.json",
    output_txt: str = "charge_distribution.txt"
) -> Dict[str, int]:
    """
    统计训练集中各罪名的样本分布
    
    Args:
        train_file: CAIL2018训练集文件路径
        output_json: 输出原始分布JSON路径
        output_txt: 输出统计报告文本路径
    
    Returns:
        Dict[str, int]: 罪名 -> 样本数量 的映射
    """
    # 流式加载数据，一行一行读，不一次性加载到内存
    # 即使1.95G也不会OOM，因为只存计数，不存整个数据集
    charge_counts: Dict[str, int] = defaultdict(int)
    total_cases = 0
    total_charge_occurrences = 0
    
    logger.info(f"Starting statistics from: {train_file}")
    
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
            
            # 获取罪名列表（支持多种CAIL2018格式）
            # 格式1: charge 在顶层（原始小样本版本）
            # 格式2: accusation 在meta下（你用的这个完整版本）
            # 格式3: charge 在meta下（比赛版本）
            charges = case.get('charge', [])
            if not charges and 'meta' in case:
                charges = case['meta'].get('accusation', [])
                if not charges:
                    charges = case['meta'].get('charge', [])
            
            # 处理空的情况
            if not charges:
                logger.warning(f"Case {line_idx} has no charges, skipping")
                continue
            
            # 计数，处理多罪名案件：每个出现的罪名都计数
            for charge in charges:
                charge_counts[charge] += 1
                total_charge_occurrences += 1
            
            total_cases += 1
            
            # 每10000个案件打印进度
            if line_idx % 10000 == 0:
                logger.info(f"Processed {line_idx} lines, found {len(charge_counts)} unique charges")
    
    # 转换为普通dict
    charge_counts = dict(charge_counts)
    
    logger.info(f"Finished processing {total_cases} cases")
    logger.info(f"Found {len(charge_counts)} unique charges")
    logger.info(f"Total charge occurrences: {total_charge_occurrences}")
    
    # 保存原始分布JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(charge_counts, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved raw distribution to {output_json}")
    
    # 生成统计报告
    report = generate_distribution_report(charge_counts, total_cases, total_charge_occurrences)
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Saved distribution report to {output_txt}")
    
    # 打印摘要到控制台
    print("\n" + "="*60)
    print("CAIL2018 罪名分布统计摘要")
    print("="*60)
    for line in report.split("\n")[:20]:
        print(line)
    print("... (full report in output file)")
    
    return charge_counts


def generate_distribution_report(
    charge_counts: Dict[str, int],
    total_cases: int,
    total_charge_occurrences: int
) -> str:
    """
    生成易读的分布统计报告
    
    Args:
        charge_counts: 罪名 -> 数量映射
        total_cases: 总案件数
        total_charge_occurrences: 总罪名单词出现次数
    
    Returns:
        str: 格式化的报告文本
    """
    # 按数量降序排序
    sorted_charges: List[Tuple[str, int]] = sorted(
        charge_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    num_charges = len(sorted_charges)
    min_count = sorted_charges[-1][1] if sorted_charges else 0
    max_count = sorted_charges[0][1] if sorted_charges else 0
    avg_count = total_charge_occurrences / num_charges if num_charges > 0 else 0
    
    # 统计分位数
    cumulative = 0
    median_count = None
    p25_count = None
    p75_count = None
    for i, (_, cnt) in enumerate(sorted_charges):
        cumulative += cnt
        ratio = cumulative / total_charge_occurrences
        if p25_count is None and ratio >= 0.25:
            p25_count = cnt
        if median_count is None and ratio >= 0.5:
            median_count = cnt
        if p75_count is None and ratio >= 0.75:
            p75_count = cnt
            break
    
    # 统计区间分布
    bins = [
        (1, 10, "1-10"),
        (11, 50, "11-50"),
        (51, 200, "51-200"),
        (201, 500, "201-500"),
        (501, 1000, "501-1000"),
        (1001, None, ">1000")
    ]
    
    bin_counts = {label: 0 for (_, _, label) in bins}
    for charge, cnt in sorted_charges:
        for low, high, label in bins:
            if high is None:
                if cnt >= low:
                    bin_counts[label] += 1
                    break
            elif low <= cnt <= high:
                bin_counts[label] += 1
                break
    
    # 构建报告
    lines = []
    lines.append("CAIL2018 罪名分布统计报告")
    lines.append("=" * 60)
    lines.append(f"总案件数: {total_cases:,}")
    lines.append(f"不同罪名总数: {num_charges:,}")
    lines.append(f"总罪名出现次数: {total_charge_occurrences:,}")
    lines.append("")
    lines.append("基本统计:")
    lines.append(f"  最少样本数: {min_count}")
    lines.append(f"  最多样本数: {max_count:,}")
    lines.append(f"  平均每个罪名: {avg_count:.1f}")
    lines.append(f"  25%分位数: {p25_count}")
    lines.append(f"  中位数: {median_count}")
    lines.append(f"  75%分位数: {p75_count}")
    lines.append("")
    lines.append("按样本数量区间分布:")
    for label, cnt in bin_counts.items():
        lines.append(f"  {label:>10}: {cnt:>4} 个罪名 ({cnt/num_charges*100:.1f}%)")
    lines.append("")
    lines.append("Top 20 样本最多的罪名:")
    lines.append("-" * 35)
    lines.append(f"{'名次':<6} {'罪名':<15} {'样本数':>8}")
    lines.append("-" * 35)
    for rank, (charge, cnt) in enumerate(sorted_charges[:20], 1):
        lines.append(f"{rank:<6} {charge:<15} {cnt:>8,}")
    lines.append("")
    lines.append("Bottom 20 样本最少的罪名:")
    lines.append("-" * 35)
    lines.append(f"{'名次':<6} {'罪名':<15} {'样本数':>8}")
    lines.append("-" * 35)
    for rank, (charge, cnt) in enumerate(reversed(sorted_charges[-20:]), 1):
        lines.append(f"{rank:<6} {charge:<15} {cnt:>8,}")
    lines.append("")
    lines.append("完整排序（罪名 -> 样本数）:")
    lines.append("-" * 35)
    for charge, cnt in sorted_charges:
        lines.append(f"  {charge:<20} {cnt:>6,}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='统计CAIL2018训练集的罪名分布，输出详细报告'
    )
    parser.add_argument(
        '--train-file',
        type=str,
        default='data/final_all_data/first_stage/train.json',
        help='CAIL2018训练集JSON文件路径（默认: data/final_all_data/first_stage/train.json）'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='data/charge_distribution.json',
        help='输出原始分布JSON路径（默认: data/charge_distribution.json）'
    )
    parser.add_argument(
        '--output-txt',
        type=str,
        default='data/charge_distribution.txt',
        help='输出统计报告文本路径（默认: data/charge_distribution.txt）'
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 运行统计
    stats_charge_distribution(
        train_file=args.train_file,
        output_json=args.output_json,
        output_txt=args.output_txt
    )


if __name__ == "__main__":
    main()
