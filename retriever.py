"""
正负案例检索器
基于embedding的相似性检索，获取与目标案件相似正例和相似错例负例

支持自适应k：
- static：传统静态公式，alpha缩放，可选归一化，保证分布均匀
"""

from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from agent import Case


@dataclass
class RetrievalResult:
    """检索结果"""
    positive_examples: List[Case]
    negative_examples: List[Case]
    positive_k: int
    negative_k: int
    max_sim_pos: float
    max_sim_neg: float
    distances: List[float]  # 对应正例的距离


class EmbeddingRetriever:
    """
    基于embedding的检索器，支持自适应k
    正例库和负例库各自独立使用一个retriever
    """
    
    def __init__(
        self, 
        embedding_model=None, 
        min_k=1, 
        max_k=5, 
        alpha=1.0, 
        normalize=False,
    ):
        """
        - min_k: 最小返回k
        - max_k: 最大返回k
        - alpha: 缩放系数，越大整体k越大
          static: alpha放大(1-sim_max)，越大k越大
        - normalize: 是否对sim_max做归一化，让分布更均匀
        """
        self.embedding_model = embedding_model
        self.case_embeddings: Optional[np.ndarray] = None
        self.cases: List[Case] = []
        self.min_k = min_k
        self.max_k = max_k
        self.alpha = alpha
        self.normalize = normalize
        self.sim_min = None  # 归一化用：所有top1最小相似度
        self.sim_max = None  # 归一化用：所有top1最大相似度
    
    def set_normalization_params(self, sim_min: float, sim_max: float):
        """设置归一化的参数，自动计算后设置"""
        self.sim_min = sim_min
        self.sim_max = sim_max
    
    def index(self, cases: List[Case], embeddings: Optional[np.ndarray] = None):
        """建立索引"""
        self.cases = cases
        if embeddings is not None:
            self.case_embeddings = embeddings
        elif self.embedding_model is not None:
            # 自动编码所有案例
            self.case_embeddings = np.array([
                self.embedding_model.encode(case.fact) for case in cases
            ])
        else:
            raise ValueError("Either embeddings or embedding_model must be provided")
        
        # L2归一化，方便余弦相似度计算
        norm = np.linalg.norm(self.case_embeddings, axis=1, keepdims=True)
        self.case_embeddings = self.case_embeddings / norm
    
    def _calculate_k_static(self, similarities: np.ndarray) -> int:
        """静态公式计算k
        公式：
          归一化开启：
            norm = (sim_max_all - sim_max) / (sim_max_all - sim_min_all) → [0,1]
            k = round(min_k + (max_k - min_k) * norm * alpha)
          归一化关闭：
            k = round(min_k + (max_k - min_k) * (1 - sim_max) * alpha)
        sim_max越大 → k越小，sim_max越小 → k越大
        """
        max_sim = similarities.max()
        
        if self.normalize and self.sim_min is not None and self.sim_max is not None:
            norm = (self.sim_max - max_sim) / (self.sim_max - self.sim_min)
            norm = max(0.0, min(1.0, norm))
        else:
            norm = (1 - max_sim)
        
        k = round(self.min_k + (self.max_k - self.min_k) * norm * self.alpha)
        k = max(self.min_k, min(self.max_k, k))
        return k
    
    def retrieve_topk(self, target_embedding: np.ndarray, k: Optional[int] = None) -> Tuple[List[Case], float, int]:
        """
        检索top-k最相似案例
        如果k不指定，则自动计算自适应k
        """
        if self.case_embeddings is None:
            raise ValueError("Call index() first")
        
        # L2归一化目标向量
        target_embedding = target_embedding / np.linalg.norm(target_embedding)
        
        # 计算余弦相似度 (越大越相似)
        similarities = self.case_embeddings @ target_embedding
        
        # 最大相似度
        max_sim = similarities.max()
        
        # 固定k直接返回
        if k is not None:
            # 获取排序索引（从大到小）
            sorted_indices = np.argsort(-similarities)
            top_indices = sorted_indices[:k]
            top_cases = [self.cases[i] for i in top_indices]
            return top_cases, max_sim, k
        
        # 自适应计算k
        k = self._calculate_k_static(similarities)
        
        # 获取排序索引（从大到小）
        sorted_indices = np.argsort(-similarities)
        top_indices = sorted_indices[:k]
        top_cases = [self.cases[i] for i in top_indices]
        
        return top_cases, max_sim, k


class AdaptiveRAGRetriever:
    """
    完整的自适应RAG检索器
    包含独立的正例检索器和负例检索器，各自计算自适应k
    """
    
    def __init__(
        self,
        pos_retriever: EmbeddingRetriever,
        neg_retriever: EmbeddingRetriever
    ):
        self.pos_retriever = pos_retriever
        self.neg_retriever = neg_retriever
    
    def retrieve(
        self,
        target_embedding: np.ndarray,
        k_positive: Optional[int] = None,
        k_negative: Optional[int] = None,
    ) -> RetrievalResult:
        """
        自适应检索正负案例
        - k_positive: 指定则固定k，None则自适应
        - k_negative: 指定则固定k，None则自适应
        """
        # 检索正例
        positive_examples, max_sim_pos, positive_k = self.pos_retriever.retrieve_topk(
            target_embedding, k_positive
        )
        
        # 检索负例
        # 对于负例，我们也是找事实相似的错例，不是找最不相似的
        # 因为我们需要"相似事实但错误判决"的案例做对比
        negative_examples, max_sim_neg, negative_k = self.neg_retriever.retrieve_topk(
            target_embedding, k_negative
        )
        
        # 计算正例距离（保持接口兼容）
        target_norm = target_embedding / np.linalg.norm(target_embedding)
        pos_embeds = self.pos_retriever.case_embeddings[
            [self.pos_retriever.cases.index(c) for c in positive_examples]
        ]
        distances = [1 - (embed @ target_norm) for embed in pos_embeds]
        
        return RetrievalResult(
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            positive_k=positive_k,
            negative_k=negative_k,
            max_sim_pos=max_sim_pos,
            max_sim_neg=max_sim_neg,
            distances=distances
        )


if __name__ == "__main__":
    # 简单测试
    print("EmbeddingRetriever 模块加载成功")
