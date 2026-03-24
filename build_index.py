"""
构建正负案例向量索引
- 正例索引：data/sampled_positives.json → data/pos_index.npy + data/pos_cases.json
- 负例索引：data/generated_negatives.json → data/neg_index.npy + data/neg_cases.json
"""

import os
import json
import numpy as np
import argparse
import logging
from typing import List

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
    logger.info(f"Loaded {len(cases)} negative examples")
    return cases


def encode_cases(cases: List[Case], model_name: str = "BAAI/bge-small-zh-v1.5") -> np.ndarray:
    """对所有case的fact做embedding"""
    # 使用国内镜像加速下载
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    model = SentenceTransformer(model_name)
    embeddings = []
    for i, case in enumerate(cases):
        embedding = model.encode(case.fact, normalize_embeddings=True)
        embeddings.append(embedding)
        if (i + 1) % 100 == 0:
            logger.info(f"Encoded {i+1}/{len(cases)}")
    return np.array(embeddings)


def save_index(cases: List[Case], embeddings: np.ndarray, output_dir: str, prefix: str):
    """保存索引：cases存json，embeddings存npy"""
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
    
    logger.info(f"Saved {len(cases)} {prefix} index to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Build embedding index for positive/negative examples')
    parser.add_argument('--pos-input', type=str, default='data/sampled_positives.json',
                       help='Input positive samples json')
    parser.add_argument('--neg-input', type=str, default='data/generated_negatives.json',
                       help='Input generated negative examples json')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for index files')
    parser.add_argument('--embedding-model', type=str, default='uer/sbert-base-chinese-nli',
                       help='Sentence embedding model name (from huggingface)')
    args = parser.parse_args()
    
    # 处理正例
    pos_cases = load_positives(args.pos_input)
    pos_embeddings = encode_cases(pos_cases, args.embedding_model)
    save_index(pos_cases, pos_embeddings, args.output_dir, "pos")
    
    # 处理负例
    neg_cases = load_negatives(args.neg_input)
    neg_embeddings = encode_cases(neg_cases, args.embedding_model)
    save_index(neg_cases, neg_embeddings, args.output_dir, "neg")
    
    logger.info("All done! Index built successfully.")
    logger.info(f"Positive index: {args.output_dir}/pos_cases.json + pos_index.npy")
    logger.info(f"Negative index: {args.output_dir}/neg_cases.json + neg_index.npy")


if __name__ == "__main__":
    main()
