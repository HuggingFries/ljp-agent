"""
正负案例检索器
基于embedding的相似性检索，获取与目标案件相似（正例）和不相似（负例）的案例

支持两种负例选择策略：
1. 随机采样：从非相似案例中随机选择
2. 最远采样：选择embedding距离最远的案例作为负例
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
    distances: List[float]  # 对应正例的距离


class EmbeddingRetriever:
    """基于embedding的正负案例检索器"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.case_embeddings: Optional[np.ndarray] = None
        self.cases: List[Case] = []
    
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
    
    def retrieve(
        self,
        target_embedding: np.ndarray,
        k_positive: int = 5,
        k_negative: int = 2,
        negative_strategy: str = "farthest"  # "farthest" or "random"
    ) -> RetrievalResult:
        """
        检索正负案例
        - k_positive: 返回k个最相似（距离最近）作为正例
        - k_negative: 返回k个不相似作为负例
        """
        if self.case_embeddings is None:
            raise ValueError("Call index() first")
        
        # L2归一化目标向量
        target_embedding = target_embedding / np.linalg.norm(target_embedding)
        
        # 计算余弦相似度 (越大越相似)
        similarities = self.case_embeddings @ target_embedding
        
        # 转换为距离 (越小越相似)
        distances = 1 - similarities
        
        # 获取排序索引
        sorted_indices = np.argsort(distances)
        
        # 正例：前k_positive个最相似
        positive_indices = sorted_indices[:k_positive]
        positive_examples = [self.cases[i] for i in positive_indices]
        positive_distances = [distances[i] for i in positive_indices]
        
        # 负例：从剩下的选
        remaining_indices = sorted_indices[k_positive:]
        
        if negative_strategy == "farthest":
            # 最远策略：选距离最远（最不相似）的作为负例
            # 从remaining中选k_negative个最远的 = 距离最大 = 在sorted末尾
            if len(remaining_indices) >= k_negative:
                negative_indices = remaining_indices[-k_negative:]
            else:
                negative_indices = remaining_indices
        elif negative_strategy == "random":
            # 随机策略
            if len(remaining_indices) >= k_negative:
                negative_indices = np.random.choice(remaining_indices, k_negative, replace=False)
            else:
                negative_indices = remaining_indices
        else:
            raise ValueError(f"Unknown negative strategy: {negative_strategy}")
        
        negative_examples = [self.cases[i] for i in negative_indices]
        
        return RetrievalResult(
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            distances=positive_distances
        )


if __name__ == "__main__":
    # 简单测试
    print("EmbeddingRetriever 模块加载成功")
