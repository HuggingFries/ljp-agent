"""
自适应正负案例检索器
支持两种检索模式：
1. flat: 全库平层检索：embedding粗筛 + LLM验证自适应k
2. hierarchical: 分层检索：罪名猜测 + embedding粗筛 + LLM验证自适应k

核心策略：
- 大模型迭代验证：判断已检索案例是否足够覆盖核心法律要素，不够就继续加，直到够了或达到max_k

Author: LJP-RAG Project
Date: 2026-04-04
"""

from typing import List, Tuple, Optional, Any, Protocol, Dict
import os
import numpy as np
import json
import logging
from dataclasses import dataclass

from agent import Case

logger = logging.getLogger(__name__)


class AdaptiveKStrategy(Protocol):
    """自适应k计算策略的接口协议
    所有具体策略都需要实现这个接口
    """
    
    def calculate_k(
        self,
        similarities: np.ndarray,
        sorted_cases: List[Case],
        target_fact: str,
    ) -> int:
        """计算最终应该返回的k值
        Args:
            similarities: 所有候选案例和目标案件的相似度数组（从大到小排序）
            sorted_cases: 排序后的候选案例列表
            target_fact: 目标案件事实文本
        Returns:
            int: 最终应该返回的k值，范围在[min_k, max_k]之间
        """
        ...


@dataclass
class LLMVerifiedAdaptiveKConfig:
    """大模型验证自适应k配置（迭代验证式，verify-update）
    
    核心思想：迭代增加案例，大模型判断当前检索是否足够回答，不够就继续加，直到够了或达到max_k
    
    Attributes:
        min_k: 最小返回案例数，少于这个必须继续加，默认=1
        max_k: 最大返回案例数，达到必须停止，控制上下文不溢出，默认=5
        initial_candidates: 第一步embedding粗筛的候选数量，从中选min_k开始迭代，默认=20
        step_add: 每次迭代增加几个候选，默认=2
    """
    min_k: int = 1
    max_k: int = 5
    initial_candidates: int = 20
    step_add: int = 2


class LLMVerifiedAdaptiveKStrategy:
    """大模型验证自适应k（迭代验证式）
    
    流程：
    1. embedding粗筛得到排序候选，先取出min_k个
    2. 让大模型判断：当前案例是否足够覆盖目标案件的核心法律要素？
    3. - [YES] 停止，返回当前k个
       - [NO] 从候选列表再拿step_add个加进去，回到步骤2
    4. 达到max_k自动停止

    """
    
    def __init__(
        self,
        config: LLMVerifiedAdaptiveKConfig,
        llm_client: Any,
        llm_model: str,
    ):
        """
        Args:
            config: 配置参数
            llm_client: OpenAI兼容的大模型客户端，需要兼容openai.ChatCompletion.create接口
            llm_model: 使用的大模型名称
        """
        self.config = config
        self.llm_client = llm_client
        self.llm_model = llm_model
    
    def _build_verify_prompt(
        self,
        target_fact: str,
        current_cases: List[Case],
    ) -> str:
        """构建验证prompt，给大模型明确判断标准"""
        cases_text = ""
        for i, case in enumerate(current_cases, 1):
            cases_text += f"### 候选案例 {i}\n"
            cases_text += f"案件事实: {case.fact[:300]}\n"  # 截断节省token
            cases_text += f"罪名: {', '.join(case.charges)}\n\n"
        
        return f"""你是一个法律检索助手，需要判断当前已检索的参考案例，是否足够对目标案件进行准确判决。

## 目标案件（需要预测判决的案件）
{target_fact}

## 当前已检索的参考案例
{cases_text}
## 请依据以下规则判断是否足够：

1. **[判断标准]** 当前已有的参考案例，是否足够帮助你对目标案件做出准确判决？
   - 如果已经有足够多的相似案例可以参考，能够支撑你做出判断 → **回答 [YES] 停止**
   - 如果仍然缺少相似案例，你觉得需要更多参考才能判断 → **回答 [NO] 继续补充**

2. **[冗余性原则]** 质量比数量重要！几个高度相似的案例比一堆不太相关的案例有用得多。如果已经有几个高度相似的案例了，保留这些就足够，不需要更多，够了就回答 [YES]

3. **[数量约束]** 
   - 当前已经选择了 {len(current_cases)} 个案例
   - 最少需要 {self.config.min_k} 个，少于最少必须回答 [NO]
   - 最多允许 {self.config.max_k} 个，达到最多必须停止并回答 [YES]

## 输出要求
最后只输出 [YES] 或者 [NO]，不要输出其他内容。

## 重要提示
- **倾向于停止**：只要你觉得已经有足够参考，就停止，不要为了"追求完美"而继续加案例
- 不需要追求"覆盖所有可能性"，有几个相似案例参考就足够判决了
- 过多冗余案例反而会干扰判决，浪费算力，够了就请及时停止
- 你已经选择了 {len(current_cases)} 个案例，超过最少要求的 {self.config.min_k} 个，满足最少数量就可以停止
"""
    
    def _parse_verification(self, response: str) -> bool:
        """解析大模型回复，True = 足够了，False = 需要更多"""
        content = response.strip().upper()
        # 匹配YES/NO
        if "[YES]" in content or "YES" in content:
            return True
        if "[NO]" in content or "NO" in content:
            return False
        # 如果没明确，默认认为够了（保守停止比继续加冗余好）
        logger.warning(f"LLM verification output unclear: {content[:50]}..., defaulting to YES")
        return True
    
    def calculate_k(
        self,
        similarities: np.ndarray,
        sorted_cases: List[Case],
        target_fact: str,
    ) -> int:
        """迭代计算最终k
        Args:
            similarities: 所有候选相似度，从大到小排序
            sorted_cases: 相似度排序后的所有案例
            target_fact: 目标案件事实
        Returns:
            int: 最终选中的案例数量，min_k ≤ k ≤ max_k
        """
        config = self.config
        # 取top initial_candidates 候选
        candidate_count = min(config.initial_candidates, len(sorted_cases))
        candidates_pool = sorted_cases[:candidate_count]
        
        # 初始选min_k个
        selected: List[Case] = candidates_pool[:config.min_k]
        remaining = candidates_pool[config.min_k:]
        
        while len(selected) < config.max_k:
            # 大模型验证是否足够
            prompt = self._build_verify_prompt(target_fact, selected)
            
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = response.choices[0].message.content.strip()
                is_enough = self._parse_verification(content)
                
                if is_enough:
                    break
                
                # 不够，加step_add个
                add_count = min(config.step_add, len(remaining), config.max_k - len(selected))
                if add_count <= 0:
                    break
                
                selected.extend(remaining[:add_count])
                remaining = remaining[add_count:]
                
            except Exception as e:
                logger.warning(f"LLM verification failed: {e}, stopping with current {len(selected)}")
                break
        
        # 最终保证在范围内
        k = len(selected)
        k = max(config.min_k, min(config.max_k, k))
        
        logger.info(f"[LLM-Verified 迭代验证] start={config.min_k}, final={k}, max={config.max_k}")
        return k


@dataclass
class FlatRetrieverConfig:
    """平层全库检索器配置
    
    Attributes:
        llm: 大模型自适应k配置
    """
    llm: LLMVerifiedAdaptiveKConfig = None


class FlatEmbeddingRetriever:
    """
    平层全库检索器，基于embedding的相似性检索 + LLM验证自适应k
    正例库和负例库各自独立使用一个retriever
    """
    
    def __init__(
        self, 
        config: FlatRetrieverConfig,
        llm_client: Any,
        llm_model: str,
        embedding_model=None,
    ):
        """
        Args:
            config: 检索器配置
            llm_client: 大模型客户端，必须提供
            llm_model: 大模型名称，必须提供
            embedding_model: sentence-transformers编码模型，None表示预编码embedding已经提供
        """
        self.config = config
        self.embedding_model = embedding_model
        self.case_embeddings: Optional[np.ndarray] = None
        self.cases: List[Case] = []
        
        # 使用默认配置如果llm为空
        if config.llm is None:
            config.llm = LLMVerifiedAdaptiveKConfig()
        
        # 总是使用LLM验证策略
        self.strategy: AdaptiveKStrategy = LLMVerifiedAdaptiveKStrategy(config.llm, llm_client, llm_model)
    
    def index(self, cases: List[Case], embeddings: Optional[np.ndarray] = None):
        """建立索引
        
        Args:
            cases: 案例列表
            embeddings: 预计算的embeddings，None则自动使用embedding_model编码
        """
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
    
    def retrieve_topk(
        self, 
        target_embedding: np.ndarray,
        target_fact: str,
        k: Optional[int] = None,
    ) -> Tuple[List[Case], float, int]:
        """检索top-k最相似案例
        如果k不指定，则使用自适应策略计算k
        
        Args:
            target_embedding: 目标案件的embedding
            target_fact: 目标案件事实文本（LLM模式需要用来打分）
            k: 固定k，None则使用自适应计算
        
        Returns:
            (top_cases, max_sim, final_k): 排序后的案例列表，最大相似度，最终k值
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
        
        # 先排序所有候选
        sorted_indices = np.argsort(-similarities)
        sorted_cases = [self.cases[i] for i in sorted_indices]
        
        # 使用LLM策略计算自适应k
        k = self.strategy.calculate_k(similarities, sorted_cases, target_fact)
        
        # 取前k个
        top_cases = sorted_cases[:k]
        
        return top_cases, max_sim, k


@dataclass
class HierarchicalRetrieverConfig:
    """分层检索器单库配置
    
    Attributes:
        index_root: 分层索引根目录
        min_k: 最少返回案例数
        max_k: 最多返回案例数
        coarse_k: 语义粗筛返回多少候选给LLM验证
        llm: 大模型自适应k配置
    """
    index_root: str = "data/index_by_charge"
    min_k: int = 1
    max_k: int = 5
    coarse_k: int = 10
    llm: LLMVerifiedAdaptiveKConfig = None


class HierarchicalEmbeddingRetriever:
    """K-Means语义聚类分层检索器
    [1. 找最近簇（根据embedding相似度）] → [2. 簇内范围过滤] → [3. 语义粗筛] → [4. LLM验证精筛]
    
    Attributes:
        config: 分层检索配置
        llm_client: 大模型客户端（给精筛用）
        llm_model: 大模型名称（给精筛用）
        prefix: "pos"表示正例，"neg"表示负例
    """
    
    def __init__(
        self,
        config: HierarchicalRetrieverConfig,
        llm_client: Any,
        llm_model: str,
        prefix: str = "pos",  # "pos" for positive, "neg" for negative
    ):
        """
        Args:
            config: 分层检索配置
            llm_client: 大模型客户端
            llm_model: 大模型名称
            prefix: 索引前缀，"pos"表示正例，"neg"表示负例
        """
        self.config = config
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.prefix = prefix
        
        # 加载簇标签列表
        self.cluster_list = self._load_cluster_list()
        # 预加载所有簇中心（用来计算相似度找最近簇）
        self.cluster_centers = self._load_cluster_centers()
        # 预加载所有案例和embeddings到内存（数据集小，没问题）
        # 一次性加载所有，按簇逻辑分组
        self.cases_map, self.embeddings_map = self._load_cluster_index()
        
        # 初始化精筛策略（始终LLM）
        self._init_strategy()
        
        logger.info(f"Loaded {prefix} clustered (K-Means) retriever:")
        logger.info(f"  Clusters: {len(self.cluster_list)}, total cases: {sum(len(v) for v in self.cases_map.values())}")
        logger.info(f"  Fine adaptive mode: llm, coarse_k={config.coarse_k}")
    
    def _load_cluster_list(self) -> List[str]:
        """加载簇标签列表"""
        cluster_list_path = os.path.join(self.config.index_root, f"{self.prefix}_cluster_list.json")
        if not os.path.exists(cluster_list_path):
            logger.warning(f"Cluster list not found: {cluster_list_path}")
            return []
        with open(cluster_list_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_cluster_centers(self) -> np.ndarray:
        """加载簇中心矩阵"""
        centers_path = os.path.join(self.config.index_root, f"{self.prefix}_cluster_centers.npy")
        if not os.path.exists(centers_path):
            logger.warning(f"Cluster centers not found: {centers_path}")
            return np.array([])
        return np.load(centers_path)
    
    def _load_cluster_index(self) -> Tuple[Dict[str, List[Case]], Dict[str, np.ndarray]]:
        """加载所有簇的索引（一次性加载，逻辑分组）
        Returns:
            (cases_map, embeddings_map): 按簇分组的案例和embedding
        """
        # 加载偏移信息
        offsets_path = os.path.join(self.config.index_root, f"{self.prefix}_cluster_offsets.json")
        cases_path = os.path.join(self.config.index_root, f"{self.prefix}_cases.json")
        index_path = os.path.join(self.config.index_root, f"{self.prefix}_index.npy")
        
        with open(offsets_path, 'r', encoding='utf-8') as f:
            cluster_offsets = json.load(f)
        
        with open(cases_path, 'r', encoding='utf-8') as f:
            all_cases_data = json.load(f)
        
        all_embeddings = np.load(index_path)
        
        # 按簇分组
        cases_map: Dict[str, List[Case]] = {}
        embeddings_map: Dict[str, np.ndarray] = {}
        
        for cluster, offset_info in cluster_offsets.items():
            start = offset_info['start']
            count = offset_info['count']
            end = start + count
            
            # 取出该簇的案例
            cases_data = all_cases_data[start:end]
            cases: List[Case] = []
            for item in cases_data:
                cases.append(Case(
                    fact=item['fact'],
                    charges=item['charges'],
                    articles=item['articles'],
                    judgment='',
                    is_positive=item['is_positive'],
                ))
            
            # 取出该簇的embeddings
            embeddings = all_embeddings[start:end]
            
            cases_map[cluster] = cases
            embeddings_map[cluster] = embeddings
        
        logger.info(f"Loaded {len(cases_map)} clusters, {len(all_cases_data)} total {self.prefix} cases")
        return cases_map, embeddings_map
    
    def _init_strategy(self):
        """初始化精筛策略（始终LLM）"""
        config = self.config
        if config.llm is None:
            config.llm = LLMVerifiedAdaptiveKConfig(
                min_k=config.min_k,
                max_k=config.max_k,
            )
        self.strategy: AdaptiveKStrategy = LLMVerifiedAdaptiveKStrategy(
            config.llm,
            self.llm_client,
            self.llm_model,
        )
    
    def retrieve_topk(
        self,
        target_embedding: np.ndarray,
        target_fact: str,
        candidate_charges: List[str] = None,
        k: Optional[int] = None,
    ) -> Tuple[List[Case], float, int]:
        """K-Means聚类分层检索top-k
        第一步：计算目标embedding和各个簇中心的相似度，取top-K最近簇
        第二步：只在最近簇内找相似案例
        
        Args:
            target_embedding: 目标案件embedding
            target_fact: 目标案件事实（给LLM验证用）
            candidate_charges: 兼容性保留，不再使用
            k: 固定k，None则自适应
        Returns:
            (top_cases, max_sim, final_k)
        """
        # 计算目标和各个簇中心的余弦相似度
        target_norm = target_embedding / np.linalg.norm(target_embedding)
        centers_norm = self.cluster_centers / np.linalg.norm(self.cluster_centers, axis=1, keepdims=True)
        similarities = centers_norm @ target_norm
        
        # 找top-2最相似簇（取两个覆盖可能性更大）
        num_clusters = len(self.cluster_list)
        top_clusters = min(2, num_clusters)
        sorted_indices = np.argsort(-similarities)
        top_cluster_indices = sorted_indices[:top_clusters]
        
        # 收集这些簇里的所有案例
        all_cases: List[Case] = []
        all_embeddings: List[np.ndarray] = []
        
        logger.info(f"[聚类分层检索] 第一层：找top-{top_clusters}最近簇，最高相似度={similarities[sorted_indices[0]]:.4f}")
        
        for idx in top_cluster_indices:
            cluster_label = self.cluster_list[idx]
            if cluster_label in self.cases_map:
                logger.info(f"[聚类分层检索] 找到簇 '{cluster_label}'，包含 {len(self.cases_map[cluster_label])} 个候选案例")
                all_cases.extend(self.cases_map[cluster_label])
                all_embeddings.append(self.embeddings_map[cluster_label])
        
        if not all_cases:
            # 如果找不到， fallback 到所有簇
            logger.warning(f"[聚类分层检索] 未找到匹配候选案例，回退到全量索引")
            for cluster in self.cluster_list:
                if cluster in self.cases_map:
                    all_cases.extend(self.cases_map[cluster])
                    all_embeddings.append(self.embeddings_map[cluster])
        
        logger.info(f"[聚类分层检索] 第一层过滤完成，剩余 {len(all_cases)} 个候选案例")
        
        if not all_cases:
            logger.error(f"[分层检索] 回退后仍然没有找到任何案例，检查索引是否构建正确，返回空结果")
            return [], 0.0, 0
        
        # 拼接所有embedding
        concatenated_emb = np.concatenate(all_embeddings, axis=0)
        # L2归一化
        target_embedding = target_embedding / np.linalg.norm(target_embedding)
        concatenated_emb = concatenated_emb / np.linalg.norm(concatenated_emb, axis=1, keepdims=True)
        
        # 计算余弦相似度
        similarities = concatenated_emb @ target_embedding
        max_sim = similarities.max()
        
        # 固定k直接返回
        if k is not None:
            sorted_indices = np.argsort(-similarities)
            top_indices = sorted_indices[:k]
            top_cases = [all_cases[i] for i in top_indices]
            logger.info(f"[分层检索] 固定k={k}，返回 {len(top_cases)} 个案例")
            return top_cases, max_sim, k
        
        # 粗筛出coarse_k候选
        sorted_indices = np.argsort(-similarities)
        coarse_k = min(self.config.coarse_k, len(sorted_indices))
        top_indices = sorted_indices[:coarse_k]
        logger.info(f"[分层检索] 第二层：embedding粗筛，选出top-{coarse_k}候选")
        
        # 精筛：用自适应策略计算最终k
        # 注意：top_indices里存的是原始索引，直接用原始索引取similarities和cases
        sorted_coarse = [(similarities[idx], all_cases[idx]) for idx in top_indices]
        # 已经按相似度从大到小排序了，因为sorted_indices就是排序后的
        # 这里再sort一遍是冗余，但不影响，保证顺序正确
        sorted_coarse.sort(key=lambda x: -x[0])
        sorted_similarities = np.array([x[0] for x in sorted_coarse])
        sorted_cases = [x[1] for x in sorted_coarse]
        
        logger.info(f"[分层检索] 第三层：自适应精筛，计算最终k")
        final_k = self.strategy.calculate_k(sorted_similarities, sorted_cases, target_fact)
        final_cases = sorted_cases[:final_k]
        
        logger.info(f"[分层检索] 完成，最终选出 {len(final_cases)} 个案例，最大相似度={max_sim:.4f}")
        
        return final_cases, max_sim, final_k


@dataclass
class RetrievalResult:
    """统一检索结果结构"""
    positive_examples: List[Case]
    negative_examples: List[Case]
    positive_k: int
    negative_k: int
    max_sim_pos: float
    max_sim_neg: float
    distances: List[float]  # 对应正例的距离（1 - 相似度）


class FlatAdaptiveRAGRetriever:
    """平层全库自适应RAG检索器
    包含独立的正例检索器和负例检索器，各自计算自适应k
    """
    
    def __init__(
        self,
        pos_retriever: FlatEmbeddingRetriever,
        neg_retriever: FlatEmbeddingRetriever
    ):
        self.pos_retriever = pos_retriever
        self.neg_retriever = neg_retriever
    
    def retrieve(
        self,
        target_embedding: np.ndarray,
        target_fact: str,
        k_positive: Optional[int] = None,
        k_negative: Optional[int] = None,
    ) -> RetrievalResult:
        """自适应检索正负案例
        - k_positive: 指定则固定k，None则自适应
        - k_negative: 指定则固定k，None则自适应
        """
        # 检索正例
        positive_examples, max_sim_pos, positive_k = self.pos_retriever.retrieve_topk(
            target_embedding, target_fact, k_positive
        )
        
        # 检索负例：我们需要"相似事实但错误判决"的案例做对比，所以同样找最相似
        negative_examples, max_sim_neg, negative_k = self.neg_retriever.retrieve_topk(
            target_embedding, target_fact, k_negative
        )
        
        # 计算正例距离（保持接口兼容）
        target_norm = target_embedding / np.linalg.norm(target_embedding)
        if self.pos_retriever.case_embeddings is not None and len(positive_examples) > 0:
            distances = [
                1 - (self.pos_retriever.case_embeddings[
                    self.pos_retriever.cases.index(c)
                ] @ target_norm)
                for c in positive_examples
            ]
        else:
            distances = []
        
        return RetrievalResult(
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            positive_k=positive_k,
            negative_k=negative_k,
            max_sim_pos=max_sim_pos,
            max_sim_neg=max_sim_neg,
            distances=distances
        )


class HierarchicalAdaptiveRAGRetriever:
    """分层自适应RAG检索器
    正例和负例都是分层索引，先预测罪名，再检索
    """
    
    def __init__(
        self,
        pos_retriever: HierarchicalEmbeddingRetriever,
        neg_retriever: HierarchicalEmbeddingRetriever,
    ):
        self.pos_retriever = pos_retriever
        self.neg_retriever = neg_retriever
    
    def retrieve(
        self,
        target_embedding: np.ndarray,
        target_fact: str,
        candidate_charges: List[str],
        k_positive: Optional[int] = None,
        k_negative: Optional[int] = None,
    ) -> RetrievalResult:
        # 检索正例
        positive_examples, max_sim_pos, positive_k = self.pos_retriever.retrieve_topk(
            target_embedding, target_fact, candidate_charges, k_positive
        )
        
        # 检索负例：同样用预测的罪名过滤
        negative_examples, max_sim_neg, negative_k = self.neg_retriever.retrieve_topk(
            target_embedding, target_fact, candidate_charges, k_negative
        )
        
        # 计算正例距离（保持接口兼容）
        target_norm = target_embedding / np.linalg.norm(target_embedding)
        # 收集所有分层索引中的正例，找到正确索引
        if len(positive_examples) > 0:
            all_pos_cases = []
            all_pos_embs_list = []
            for charge in self.pos_retriever.cases_map:
                if charge in self.pos_retriever.cases_map:
                    all_pos_cases.extend(self.pos_retriever.cases_map[charge])
                    all_pos_embs_list.append(self.pos_retriever.embeddings_map[charge])
            if all_pos_embs_list:
                all_pos_embs = np.concatenate(all_pos_embs_list, axis=0)
                distances = []
                for case in positive_examples:
                    idx = all_pos_cases.index(case)
                    embed = all_pos_embs[idx]
                    distances.append(1 - (embed @ target_norm))
            else:
                distances = []
        else:
            distances = []
        
        return RetrievalResult(
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            positive_k=positive_k,
            negative_k=negative_k,
            max_sim_pos=max_sim_pos,
            max_sim_neg=max_sim_neg,
            distances=distances,
        )


def load_retriever_config(config_path: str = "config.json") -> dict:
    """从配置文件加载retriever配置
    方便统一管理所有超参数
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get("retriever", {})


def create_retriever_from_config(
    config_dict: dict,
    llm_client: Any,
    llm_model: str,
) -> any:
    """根据配置字典创建检索器
    支持两种模式：flat / hierarchical
    
    Args:
        config_dict: 配置字典，从json加载
        llm_client: 大模型客户端，必须提供
        llm_model: 大模型名称，必须提供
    
    Returns:
        检索器实例：FlatAdaptiveRAGRetriever 或 HierarchicalAdaptiveRAGRetriever
    """
    mode = config_dict.get("mode", "hierarchical")
    
    if mode == "flat":
        # 平层检索模式
        def _parse_flat_config(cfg: dict) -> FlatRetrieverConfig:
            """解析单个平层检索器配置"""
            llm_config = LLMVerifiedAdaptiveKConfig(
                min_k=cfg.get("min_k", 1),
                max_k=cfg.get("max_k", 5),
                initial_candidates=cfg.get("initial_candidates", 20),
                step_add=cfg.get("step_add", 2),
            )
            return FlatRetrieverConfig(
                llm=llm_config,
            )
        
        # 解析正负例配置
        positive_dict = config_dict.get("positive", {})
        negative_dict = config_dict.get("negative", {})
        
        pos_config = _parse_flat_config(positive_dict)
        neg_config = _parse_flat_config(negative_dict)
        
        # 加载正例索引数据（使用index_clustered下的全量2000+案例，和分层保持一致）
        from agent import Case
        import json
        import numpy as np
        index_root = "data/index_clustered"
        pos_cases_path = f"{index_root}/pos_cases.json"
        pos_embeddings_path = f"{index_root}/pos_index.npy"
        with open(pos_cases_path, 'r', encoding='utf-8') as f:
            pos_case_dicts = json.load(f)
        pos_cases = [
            Case(
                fact=c.get("fact", ""),
                charges=c.get("charges", []),
                articles=c.get("articles", []),
                judgment=c.get("judgment", ""),
                is_positive=c.get("is_positive", True)
            )
            for c in pos_case_dicts
        ]
        pos_embeddings = np.load(pos_embeddings_path)
        
        pos_retriever = FlatEmbeddingRetriever(
            config=pos_config,
            llm_client=llm_client,
            llm_model=llm_model,
        )
        pos_retriever.index(pos_cases, pos_embeddings)
        
        # 加载负例索引数据
        neg_cases_path = f"{index_root}/neg_cases.json"
        neg_embeddings_path = f"{index_root}/neg_index.npy"
        with open(neg_cases_path, 'r', encoding='utf-8') as f:
            neg_case_dicts = json.load(f)
        neg_cases = [
            Case(
                fact=c.get("fact", ""),
                charges=c.get("charges", []),
                articles=c.get("articles", []),
                judgment=c.get("judgment", ""),
                is_positive=c.get("is_positive", False)
            )
            for c in neg_case_dicts
        ]
        neg_embeddings = np.load(neg_embeddings_path)
        
        neg_retriever = FlatEmbeddingRetriever(
            config=neg_config,
            llm_client=llm_client,
            llm_model=llm_model,
        )
        neg_retriever.index(neg_cases, neg_embeddings)
        
        logger.info(f"[Flat检索初始化完成] 正例: {len(pos_cases)} cases, 负例: {len(neg_cases)} cases")
        return FlatAdaptiveRAGRetriever(pos_retriever, neg_retriever)
    
    elif mode == "hierarchical":
        # 分层检索模式：K-Means语义聚类，索引存在data/index_clustered
        logger.info("[初始化] 使用K-Means语义聚类分层检索模式")
        def _parse_hierarchical_config(cfg: dict, index_root: str) -> HierarchicalRetrieverConfig:
            """解析单个分层检索器配置"""
            llm_config = LLMVerifiedAdaptiveKConfig(
                min_k=cfg.get("min_k", 1),
                max_k=cfg.get("max_k", 5),
                initial_candidates=cfg.get("initial_candidates", 20),
                step_add=cfg.get("step_add", 2),
            )
            return HierarchicalRetrieverConfig(
                index_root=index_root,
                min_k=cfg.get("min_k", 1),
                max_k=cfg.get("max_k", 5),
                coarse_k=config_dict.get("coarse_k", 10),
                llm=llm_config,
            )
        
        # K-Means聚类索引固定输出到data/index_clustered
        index_root = "data/index_clustered"
        
        positive_dict = config_dict.get("positive", {})
        negative_dict = config_dict.get("negative", {})
        
        pos_config = _parse_hierarchical_config(positive_dict, index_root)
        neg_config = _parse_hierarchical_config(negative_dict, index_root)
        
        pos_retriever = HierarchicalEmbeddingRetriever(
            config=pos_config,
            llm_client=llm_client,
            llm_model=llm_model,
            prefix="pos",
        )
        
        neg_retriever = HierarchicalEmbeddingRetriever(
            config=neg_config,
            llm_client=llm_client,
            llm_model=llm_model,
            prefix="neg",
        )
        
        return HierarchicalAdaptiveRAGRetriever(pos_retriever, neg_retriever)
    
    else:
        raise ValueError(f"Unknown retriever mode: {mode}, must be 'flat' or 'hierarchical'")


if __name__ == "__main__":
    # 简单测试
    print("Adaptive Retriever 模块加载成功，支持两种检索模式：")
    print("  - flat: 全库平层检索 → embedding粗筛 + LLM验证自适应k")
    print("  - hierarchical: 分层检索 → 罪名过滤 → embedding粗筛 + LLM验证自适应k")
    print("核心策略：仅保留LLM驱动迭代验证自适应k")
