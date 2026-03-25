"""
分析案例embedding的余弦相似度分布
用于确定自适应k的参数范围
"""

import json
import numpy as np


def main(min_k=1, max_k=5, alpha=1.0, normalize=False):
    """读取正负案例embeddings，分析相似度分布，预测自适应k分布
    alpha: 缩放系数，>1会放大(1-sim_max)，让k整体变大，增加大k出现频率
    normalize: 是否对sim_max做归一化，让k分布更均匀
    """
    # 读取正负案例
    with open('data/pos_cases.json', 'r', encoding='utf-8') as f:
        pos = json.load(f)
    with open('data/neg_cases.json', 'r', encoding='utf-8') as f:
        neg = json.load(f)
    all_cases = pos + neg
    
    # 加载embeddings
    pos_embeddings = np.load('data/pos_index.npy', allow_pickle=False)
    neg_embeddings = np.load('data/neg_index.npy', allow_pickle=False)
    all_embeddings = np.vstack([pos_embeddings, neg_embeddings])
    
    print(f"总案例数: {len(all_cases)}")
    print(f"Embedding shape: {all_embeddings.shape}")
    print(f"Alpha缩放系数: {alpha}")
    print(f"Normalize归一化: {'ON' if normalize else 'OFF'}")
    
    # 计算两两余弦相似度
    norm = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norm_emb = all_embeddings / norm
    sim_matrix = norm_emb @ norm_emb.T
    
    # 统计对角线外的相似度分布（排除自己和自己）
    sim_values = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)].flatten()
    
    # 统计分布
    print("\n=== 两两相似度分布统计 ===")
    print(f"最小相似度: {sim_values.min():.4f}")
    print(f"最大相似度: {sim_values.max():.4f}")
    print(f"均值: {sim_values.mean():.4f}")
    print(f"中位数: {np.median(sim_values):.4f}")
    print(f"标准差: {sim_values.std():.4f}")
    print("\n分位数:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        print(f"  {q*100:>4}%: {np.quantile(sim_values, q):.4f}")
    
    # 统计每个案例的最大相似度（top1相似度）
    sim_max_list = []
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix[i].copy()
        row[i] = 0  # 排除自己
        sim_max = row.max()
        sim_max_list.append(sim_max)
    
    sim_min = np.min(sim_max_list)
    sim_max = np.max(sim_max_list)
    print(f"\n=== Top1相似度（每个案例的最大相似度）===")
    print(f"min sim_max: {sim_min:.4f}")
    print(f"max sim_max: {sim_max:.4f}")
    print(f"mean sim_max: {np.mean(sim_max_list):.4f}")
    print(f"median sim_max: {np.median(sim_max_list):.4f}")
    
    # 计算自适应k分布
    k_list = []
    if normalize:
        # 归一化版本，让分布更均匀
        # sim_max越大 → k越小，所以归一化后：
        # (sim_max_all - sim_max) / (sim_max_all - sim_min_all) → [0,1]
        # sim_max最大 → 0 → k最小
        # sim_max最小 → 1 → k最大
        sim_max_all = sim_max
        sim_min_all = sim_min
        for sim_max in sim_max_list:
            norm = (sim_max_all - sim_max) / (sim_max_all - sim_min_all)
            norm = max(0.0, min(1.0, norm))
            k = round(min_k + (max_k - min_k) * norm * alpha)
            k = max(min_k, min(max_k, k))
            k_list.append(k)
    else:
        # 原始版本，带alpha缩放，就是最初版本
        for sim_max in sim_max_list:
            k = round(min_k + (max_k - min_k) * (1 - sim_max) * alpha)
            k = max(min_k, min(max_k, k))
            k_list.append(k)
    
    print(f"\n=== 自适应k分布（min_k={min_k}, max_k={max_k}, alpha={alpha}, normalize={normalize}）===")
    print(f"k分布:")
    for k in range(min_k, max_k+1):
        cnt = k_list.count(k)
        print(f"  k={k}: {cnt} 个案例 ({cnt/len(k_list)*100:.1f}%)")
    
    # 输出每个k对应的top1相似度临界值
    print(f"\n=== 自适应k临界值（top1相似度范围）===")
    if normalize:
        delta = sim_max - sim_min
        # 对于每个k，找出sim_max临界值
        # k = round(min_k + (max_k - min_k) * ((sim_max - s) / delta) * alpha)
        # 反推s：
        # k - 0.5 = min_k + (max_k - min_k) * ((sim_max - s) / delta) * alpha
        # => s = sim_max - ((k - 0.5 - min_k) * delta) / ((max_k - min_k) * alpha)
        print(f"{'k':<5} {'sim_max >':<12} 范围说明")
        print("-" * 35)
        prev_critical = sim_max + 0.0001
        for k in range(min_k, max_k + 1):
            if k < max_k:
                critical = sim_max - ((k + 0.5 - min_k) * delta) / ((max_k - min_k) * alpha)
                print(f"k={k:<3} {'>':<2} {critical:<10.4f}  ({critical:.4f} < sim_max ≤ {prev_critical:.4f})")
                prev_critical = critical
            else:
                # 最后一个k
                critical = sim_max - ((k - 0.5 - min_k) * delta) / ((max_k - min_k) * alpha)
                print(f"k={k:<3}    ≤ {critical:<10.4f}  (sim_max ≤ {critical:.4f})")
    else:
        # 非归一化版本，sim_max是原始相似度
        # k = round(min_k + (max_k - min_k) * (1 - sim_max) * alpha)
        print(f"{'k':<5} {'sim_max >':<12} 范围说明")
        print("-" * 35)
        prev_critical = 1.0001
        for k in range(min_k, max_k + 1):
            if k < max_k:
                critical = 1 - ((k + 0.5 - min_k)) / ((max_k - min_k) * alpha)
                print(f"k={k:<3} {'>':<2} {critical:<10.4f}  ({critical:.4f} < sim_max ≤ {prev_critical:.4f})")
                prev_critical = critical
            else:
                critical = 1 - ((k - 0.5 - min_k)) / ((max_k - min_k) * alpha)
                print(f"k={k:<3}    ≤ {critical:<10.4f}  (sim_max ≤ {critical:.4f})")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze similarity distribution and adaptive k distribution')
    parser.add_argument('--min-k', type=int, default=1, help='Minimum k for adaptive')
    parser.add_argument('--max-k', type=int, default=5, help='Maximum k for adaptive')
    parser.add_argument('--alpha', type=float, default=1.0, help='Scaling factor for (1-sim_max), larger = larger k')
    parser.add_argument('--normalize', action='store_true', help='Normalize sim_max to get more uniform k distribution')
    args = parser.parse_args()
    main(args.min_k, args.max_k, args.alpha, args.normalize)
