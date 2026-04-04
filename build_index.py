"""
构建正负案例向量索引，支持三种模式：
1. flat: 传统平面索引，所有案例放一起（兼容旧版本）
2. clustered: K-Means语义聚类分层索引，按语义分簇，检索先找簇再找相似（推荐）
   - 解决按罪名分组缺点：事实相似但罪名不同的案例会被错误过滤

聚类索引构建流程：
- 所有案例embedding做K-Means聚类，每个案例得到一个簇标签
- 按簇逻辑分组存储，加载时一次性读入，检索第一步找最相似簇，再在簇内找相似
"""

import os
import json
import numpy as np
import argparse
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

# 使用国内镜像加速huggingface下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sklearn.cluster import KMeans

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
    负例保存true_charges方便分组，但聚类还是用事实embedding
    """
    cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            # 负例保持wrong标签
            case = Case(
                fact=item['fact'],
                charges=item['wrong_charges'],
                articles=item['wrong_articles'],
                judgment='',
                is_positive=False
            )
            # 保存正确罪名，可选保留但不用于分组
            case.true_charges = item['true_charges']
            cases.append(case)
    logger.info(f"Loaded {len(cases)} negative examples")
    return cases


def encode_cases(cases: List[Case], model: SentenceTransformer) -> np.ndarray:
    """对所有case的fact做embedding"""
    embeddings = []
    for i, case in enumerate(cases):
        embedding = model.encode(case.fact, normalize_embeddings=True)
        embeddings.append(embedding)
        if (i + 1) % 100 == 0:
            logger.info(f"Encoded {i+1}/{len(cases)}")
    return np.array(embeddings)


def do_kmeans_clustering(embeddings: np.ndarray, n_clusters: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """对embeddings做K-Means聚类
    Args:
        embeddings: (n_samples, n_dim) 所有案例的embedding
        n_clusters: 聚类数目k
    Returns:
        labels: 每个案例的簇标签
        cluster_centers: 簇中心 (n_clusters, n_dim)
    """
    logger.info(f"Running K-Means clustering with n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    
    # 统计每个簇大小
    cluster_counts = np.bincount(labels)
    for i in range(n_clusters):
        logger.info(f"  Cluster {i}: {cluster_counts[i]} cases")
    
    return labels, cluster_centers


def save_clustered_index(
    cases: List[Case],
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_centers: np.ndarray,
    output_root: str,
    prefix: str,  # "pos" or "neg"
    n_clusters: int,
):
    """保存聚类分层索引：统一保存在几个文件中，按簇逻辑分组
    - {prefix}_cases.json: 所有案例列表，每个案例带cluster标签
    - {prefix}_index.npy: 所有embeddings拼接在一起
    - {prefix}_cluster_offsets.json: 每个簇在大数组中的起始偏移和长度
    - {prefix}_cluster_centers.npy: 簇中心矩阵，用来在线找最近簇
    - {prefix}_cluster_list.json: 所有簇标签列表
    """
    os.makedirs(output_root, exist_ok=True)
    
    # 收集所有案例，记录每个簇的偏移
    all_cases = []
    all_embeddings = []
    cluster_offsets = {}
    current_offset = 0
    
    # 按簇顺序拼接
    clusters = list(range(n_clusters))
    
    for label in clusters:
        # 找出所有该簇的案例和embedding
        indices = np.where(labels == label)[0]
        n = len(indices)
        
        if n == 0:
            logger.warning(f"Cluster {label} is empty, skipping")
            continue
        
        # 记录偏移信息
        cluster_offsets[str(label)] = {  # json key必须是字符串
            "start": current_offset,
            "count": n
        }
        
        # 添加到总列表
        for idx in indices:
            case = cases[idx]
            all_cases.append({
                "fact": case.fact,
                "charges": case.charges,
                "articles": case.articles,
                "is_positive": case.is_positive,
                "cluster": int(label),
            })
        
        # 添加embedding
        all_embeddings.append(embeddings[indices])
        current_offset += n
    
    # 拼接所有embeddings
    concatenated = np.concatenate(all_embeddings, axis=0)
    
    # 保存簇标签列表
    cluster_list = list(cluster_offsets.keys())
    with open(os.path.join(output_root, f"{prefix}_cluster_list.json"), 'w', encoding='utf-8') as f:
        json.dump(cluster_list, f, ensure_ascii=False, indent=2)
    
    # 保存偏移信息
    with open(os.path.join(output_root, f"{prefix}_cluster_offsets.json"), 'w', encoding='utf-8') as f:
        json.dump(cluster_offsets, f, ensure_ascii=False, indent=2)
    
    # 保存簇中心
    np.save(os.path.join(output_root, f"{prefix}_cluster_centers.npy"), cluster_centers)
    
    # 保存所有案例
    with open(os.path.join(output_root, f"{prefix}_cases.json"), 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)
    
    # 保存所有embeddings
    np.save(os.path.join(output_root, f"{prefix}_index.npy"), concatenated)
    
    total_cases = len(all_cases)
    logger.info(f"Saved clustered {prefix} index: {len(cluster_offsets)} clusters, {total_cases} total cases")
    logger.info(f"  Files:")
    logger.info(f"  ├─ {prefix}_cluster_list.json    : 所有簇标签列表")
    logger.info(f"  ├─ {prefix}_cluster_offsets.json: 每个簇偏移信息")
    logger.info(f"  ├─ {prefix}_cluster_centers.npy : 簇中心矩阵")
    logger.info(f"  ├─ {prefix}_cases.json         : 所有案例")
    logger.info(f"  └─ {prefix}_index.npy          : 所有embeddings")


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
    parser = argparse.ArgumentParser(description='Build embedding index, support flat/clustered')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, default='clustered', choices=['flat', 'clustered'],
                       help='Index mode: flat=all in one, clustered=K-Means semantic clustering')
    parser.add_argument('--pos-input', type=str, default='data/sampled_positives.json',
                       help='Input positive samples json')
    parser.add_argument('--neg-input', type=str, default='data/generated_negatives.json',
                       help='Input generated negative examples json')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for index files (overrides config)')
    parser.add_argument('--embedding-model', type=str, default=None,
                       help='Sentence embedding model name (overrides config)')
    parser.add_argument('--n-clusters', type=int, default=None,
                       help='Number of clusters for K-Means (overrides config, default: based on total cases)')
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
        # ========== 平面索引 ==========
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
        # ========== K-Means聚类分层索引 ==========
        # 处理正例
        pos_cases = load_positives(args.pos_input)
        logger.info(f"Encoding all positive cases...")
        pos_embeddings = encode_cases(pos_cases, model)
        
        # 处理负例
        neg_cases = load_negatives(args.neg_input)
        logger.info(f"Encoding all negative cases...")
        neg_embeddings = encode_cases(neg_cases, model)
        
        # 确定聚类数目k
        total_pos = len(pos_cases)
        if args.n_clusters is not None:
            n_clusters_pos = args.n_clusters
        else:
            # 经验规则：平均每个簇大概20-40个案例
            n_clusters_pos = max(10, total_pos // 30)
        logger.info(f"Positive total: {total_pos}, n_clusters={n_clusters_pos} (avg {total_pos/n_clusters_pos:.1f} per cluster)")
        
        # K-Means聚类正例
        pos_labels, pos_centers = do_kmeans_clustering(pos_embeddings, n_clusters_pos)
        
        # 负例聚类（独立聚类）
        total_neg = len(neg_cases)
        if args.n_clusters is not None:
            n_clusters_neg = args.n_clusters
        else:
            n_clusters_neg = max(10, total_neg // 30)
        logger.info(f"Negative total: {total_neg}, n_clusters={n_clusters_neg} (avg {total_neg/n_clusters_neg:.1f} per cluster)")
        
        neg_labels, neg_centers = do_kmeans_clustering(neg_embeddings, n_clusters_neg)
        
        # 保存聚类索引
        clustered_root = os.path.join(output_dir, "index_clustered")
        save_clustered_index(pos_cases, pos_embeddings, pos_labels, pos_centers, clustered_root, "pos", n_clusters_pos)
        save_clustered_index(neg_cases, neg_embeddings, neg_labels, neg_centers, clustered_root, "neg", n_clusters_neg)
        
        logger.info("All done! Clustered (K-Means semantic) index built successfully.")
        logger.info(f"Index root: {clustered_root}")


if __name__ == "__main__":
    main()
