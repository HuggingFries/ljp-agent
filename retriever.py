"""
正负案例检索器
基于embedding的相似性检索，获取与目标案件相似正例和相似错例负例

支持多种自适应k策略：
- static: 传统静态数学公式计算，alpha缩放，可选归一化，保证分布均匀
- llm: 大模型评估法律相关性，自决定k，只保留真正相关的案例

Author: Your Name
Date: 2026-03-26
"""

from typing import List, Tuple, Optional, Any, Protocol
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
            similarities: 所有候选案例和目标案例的相似度数组（从大到小排序）
            sorted_cases: 排序后的候选案例列表
            target_fact: 目标案件事实文本
        Returns:
            int: 最终应该返回的k值，范围在[min_k, max_k]之间
        """
        ...


@dataclass
class StaticAdaptiveKConfig:
    """静态数学公式自适应k配置
    
    Attributes:
        min_k: 最小返回案例数，默认=1
        max_k: 最大返回案例数，默认=5
        alpha: 缩放系数，越大整体k越大。公式: k = round(min_k + (max_k - min_k) * (1 - sim_max) * alpha)，默认=1.0
        normalize: 是否对sim_max做归一化，让k分布更均匀。如果数据集相似度分布偏移，开启后效果更好，默认=False
        sim_min: 归一化用，所有top1相似度的最小值，None则自动计算，默认=None
        sim_max: 归一化用，所有top1相似度的最大值，None则自动计算，默认=None
    """
    min_k: int = 1
    max_k: int = 5
    alpha: float = 1.0
    normalize: bool = False
    sim_min: Optional[float] = None
    sim_max: Optional[float] = None


class StaticAdaptiveKStrategy:
    """静态数学公式自适应k
    核心思想：sim_max越大（已经有高相似案例）→ k越小；sim_max越小 → k越大
    """
    
    def __init__(self, config: StaticAdaptiveKConfig):
        self.config = config
        self._sim_min = config.sim_min
        self._sim_max = config.sim_max
    
    def set_normalization_params(self, sim_min: float, sim_max: float):
        """设置归一化参数，一般在构建索引后自动计算设置"""
        self._sim_min = sim_min
        self._sim_max = sim_max
    
    def calculate_k(
        self,
        similarities: np.ndarray,
        sorted_cases: List[Case],
        target_fact: str,
    ) -> int:
        max_sim = similarities.max()
        
        if self.config.normalize and self._sim_min is not None and self._sim_max is not None:
            norm = (self._sim_max - max_sim) / (self._sim_max - self._sim_min)
            norm = max(0.0, min(1.0, norm))
        else:
            norm = (1 - max_sim)
        
        k = round(self.config.min_k + (self.config.max_k - self.config.min_k) * norm * self.config.alpha)
        k = max(self.config.min_k, min(self.config.max_k, k))
        return k


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
    
    优点：
    - 比逐个打分少很多调用（平均 2-3次vs 10次），节省算力
    - 大模型基于整体判断，更准确；明确给出判断标准，减少模糊性
    - 严格保证min_k ≤ k ≤ max_k，不会出现信息不足
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

1. **[必要条件]** 必须覆盖目标案件的**核心争议焦点**和**法律构成要件**：
   - 如果目标案件的关键事实（侵犯客体、客观行为、主观要件）在现有检索案例中没有类似情况 → 必须回答 [NO]，需要补充更多案例
   - 如果核心法律要素已经被覆盖 → 可以回答 [YES]

2. **[冗余性]** 如果多个案例讲的是同一个法律要点，保留少量就够了，不需要重复。已经够了就回答 [YES]

3. **[数量约束]** 
   - 当前已经选择了 {len(current_cases)} 个案例
   - 最少需要 {self.config.min_k} 个，少于最少必须回答 [NO]
   - 最多允许 {self.config.max_k} 个，达到最多必须停止并回答 [YES]

## 输出要求
最后只输出 [YES] 或者 [NO]，不要输出其他内容。

## 重要提示
- 不需要追求"完全覆盖"，只要目标案件的**核心法律要素**已经有类似案例参考就足够了
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
class EmbeddingRetrieverConfig:
    """Embedding检索器配置
    
    Attributes:
        adaptive_mode: 自适应k模式，可选值: "static" | "llm"，默认="static"
        static_config: static模式的配置，adaptive_mode="static"时生效
        llm_config: llm模式的配置，adaptive_mode="llm"时生效
    """
    adaptive_mode: str = "static"  # "static" or "llm"
    static_config: StaticAdaptiveKConfig = None
    llm_config: LLMVerifiedAdaptiveKConfig = None


class EmbeddingRetriever:
    """
    基于embedding的检索器，支持多种自适应k策略
    正例库和负例库各自独立使用一个retriever
    """
    
    def __init__(
        self, 
        embedding_model=None,
        config: Optional[EmbeddingRetrieverConfig] = None,
        llm_client: Optional[Any] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Args:
            embedding_model: sentence-transformers编码模型，None表示预编码embedding已经提供
            config: 检索器配置，包含自适应模式选择
            llm_client: 大模型客户端，adaptive_mode="llm"时需要
            llm_model: 大模型名称，adaptive_mode="llm"时需要
        """
        self.embedding_model = embedding_model
        self.case_embeddings: Optional[np.ndarray] = None
        self.cases: List[Case] = []
        
        # 使用默认配置如果没提供
        if config is None:
            config = EmbeddingRetrieverConfig()
        
        self.config = config
        
        # 根据模式初始化策略
        if config.adaptive_mode == "static":
            if config.static_config is None:
                config.static_config = StaticAdaptiveKConfig()
            self.strategy: AdaptiveKStrategy = StaticAdaptiveKStrategy(config.static_config)
        elif config.adaptive_mode == "llm":
            if config.llm_config is None:
                config.llm_config = LLMVerifiedAdaptiveKConfig()
            if llm_client is None or llm_model is None:
                raise ValueError("LLM mode requires llm_client and llm_model")
            self.strategy: AdaptiveKStrategy = LLMVerifiedAdaptiveKStrategy(config.llm_config, llm_client, llm_model)
        else:
            raise ValueError(f"Unknown adaptive_mode: {config.adaptive_mode}, expected 'static' or 'llm'")
    
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
        
        # 如果是static模式且开启归一化，自动计算sim_min/sim_max
        if (
            self.config.adaptive_mode == "static"
            and self.config.static_config is not None
            and self.config.static_config.normalize
        ):
            static_config = self.config.static_config
            if (static_config.sim_min is None or static_config.sim_max is None) and self.case_embeddings is not None:
                norm_emb = self.case_embeddings
                sim_matrix = norm_emb @ norm_emb.T
                sim_max_list = []
                for i in range(sim_matrix.shape[0]):
                    row = sim_matrix[i].copy()
                    row[i] = 0
                    sim_max_list.append(row.max())
                auto_sim_min = np.min(sim_max_list)
                auto_sim_max = np.max(sim_max_list)
                sim_min = static_config.sim_min if static_config.sim_min is not None else float(auto_sim_min)
                sim_max = static_config.sim_max if static_config.sim_max is not None else float(auto_sim_max)
                if isinstance(self.strategy, StaticAdaptiveKStrategy):
                    self.strategy.set_normalization_params(sim_min, sim_max)
                logger.info(f"Auto computed normalization params: sim_min={sim_min:.4f}, sim_max={sim_max:.4f}")
    
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
        
        # 使用策略计算自适应k
        k = self.strategy.calculate_k(similarities, sorted_cases, target_fact)
        
        # 取前k个
        top_cases = sorted_cases[:k]
        
        return top_cases, max_sim, k


@dataclass
class AdaptiveRAGRetrieverConfig:
    """完整自适应RAG检索器配置
    
    Attributes:
        positive_config: 正例检索器配置
        negative_config: 负例检索器配置
    """
    positive_config: EmbeddingRetrieverConfig
    negative_config: EmbeddingRetrieverConfig


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
        
        # 检索负例
        # 对于负例，我们也是找事实相似的错例，不是找最不相似的
        # 因为我们需要"相似事实但错误判决"的案例做对比
        negative_examples, max_sim_neg, negative_k = self.neg_retriever.retrieve_topk(
            target_embedding, target_fact, k_negative
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


def load_retriever_config(config_path: str = "config.json") -> dict:
    """从配置文件加载retriever配置
    方便统一管理所有超参数
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get("retriever", {})


def create_retriever_from_config(
    config_dict: dict,
    llm_client: Optional[Any] = None,
    llm_model: Optional[str] = None,
) -> AdaptiveRAGRetriever:
    """根据配置字典创建检索器
    
    Args:
        config_dict: 配置字典，从json加载
        llm_client: 大模型客户端，llm模式需要
        llm_model: 大模型名称，llm模式需要
    
    Returns:
        AdaptiveRAGRetriever: 配置好的检索器
    """
    def _parse_embedding_config(cfg: dict) -> EmbeddingRetrieverConfig:
        """解析单个检索器配置"""
        adaptive_mode = cfg.get("adaptive_mode", "static")
        
        static_config = None
        if adaptive_mode == "static":
            static_kwargs = cfg.get("static", {})
            static_config = StaticAdaptiveKConfig(**static_kwargs)
        
        llm_config = None
        if adaptive_mode == "llm":
            llm_kwargs = cfg.get("llm", {})
            llm_config = LLMVerifiedAdaptiveKConfig(**llm_kwargs)
        
        return EmbeddingRetrieverConfig(
            adaptive_mode=adaptive_mode,
            static_config=static_config,
            llm_config=llm_config,
        )
    
    # 解析正负例配置
    positive_dict = config_dict.get("positive", {})
    negative_dict = config_dict.get("negative", {})
    
    pos_config = _parse_embedding_config(positive_dict)
    neg_config = _parse_embedding_config(negative_dict)
    
    # 创建检索器
    pos_retriever = EmbeddingRetriever(
        embedding_model=None,
        config=pos_config,
        llm_client=llm_client,
        llm_model=llm_model,
    )
    
    neg_retriever = EmbeddingRetriever(
        embedding_model=None,
        config=neg_config,
        llm_client=llm_client,
        llm_model=llm_model,
    )
    
    return AdaptiveRAGRetriever(pos_retriever, neg_retriever)


@dataclass
class HierarchicalRetrieverConfig:
    """分层检索器配置
    
    Attributes:
        index_root: 分层索引根目录
        embedding_model_name: embedding模型名称
        min_k: 最少返回案例数
        max_k: 最多返回案例数
        coarse_k: 语义粗筛返回多少候选给LLM验证
        adaptive_mode: 最终精筛的自适应模式: "static" | "llm"
        static_config: static模式配置
        llm_config: llm模式配置
    """
    index_root: str = "data/index_by_charge"
    embedding_model_name: str = "uer/sbert-base-chinese-nli"
    min_k: int = 1
    max_k: int = 5
    coarse_k: int = 10
    adaptive_mode: str = "llm"
    static_config: StaticAdaptiveKConfig = None
    llm_config: LLMVerifiedAdaptiveKConfig = None


class HierarchicalEmbeddingRetriever:
    """分层检索器
    [1. 罪名候选] → [2. 范围过滤] → [3. 语义粗筛] → [4. LLM验证精筛]
    
    Attributes:
        config: 分层检索配置
        llm_client: 大模型客户端（给精筛用）
        llm_model: 大模型名称（给精筛用）
    """
    
    def __init__(
        self,
        config: HierarchicalRetrieverConfig,
        llm_client: Optional[Any] = None,
        llm_model: Optional[str] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.llm_model = llm_model
        
        # 加载罪名列表
        self.pos_charge_list = self._load_charge_list("pos")
        self.neg_charge_list = self._load_charge_list("neg")
        
        # 预加载所有案例和embeddings到内存（数据集小，没问题）
        self.pos_cases: Dict[str, List[Case]] = {}
        self.pos_embeddings: Dict[str, np.ndarray] = {}
        for charge in self.pos_charge_list:
            cases, embeddings = self._load_charge_index("pos", charge)
            self.pos_cases[charge] = cases
            self.pos_embeddings[charge] = embeddings
        
        self.neg_cases: Dict[str, List[Case]] = {}
        self.neg_embeddings: Dict[str, np.ndarray] = {}
        for charge in self.neg_charge_list:
            cases, embeddings = self._load_charge_index("neg", charge)
            self.neg_cases[charge] = cases
            self.neg_embeddings[charge] = embeddings
        
        # 初始化精筛策略
        self._init_strategy()
        
        logger.info(f"Loaded hierarchical retriever:")
        logger.info(f"  Positive charges: {len(self.pos_charge_list)}, total cases: {sum(len(v) for v in self.pos_cases.values())}")
        logger.info(f"  Negative charges: {len(self.neg_charge_list)}, total cases: {sum(len(v) for v in self.neg_cases.values())}")
        logger.info(f"  Fine adaptive mode: {config.adaptive_mode}, coarse_k={config.coarse_k}")
    
    def _load_charge_list(self, prefix: str) -> List[str]:
        """加载罪名列表"""
        charge_list_path = os.path.join(self.config.index_root, f"{prefix}_charge_list.json")
        if not os.path.exists(charge_list_path):
            logger.warning(f"Charge list not found: {charge_list_path}")
            return []
        with open(charge_list_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_charge_index(self, prefix: str, charge: str) -> Tuple[List[Case], np.ndarray]:
        """加载单个罪名的索引"""
        safe_charge = charge.replace('/', '_')
        charge_dir = os.path.join(self.config.index_root, prefix, safe_charge)
        cases_path = os.path.join(charge_dir, "cases.json")
        index_path = os.path.join(charge_dir, "index.npy")
        
        with open(cases_path, 'r', encoding='utf-8') as f:
            cases_data = json.load(f)
        
        cases: List[Case] = []
        for item in cases_data:
            cases.append(Case(
                fact=item['fact'],
                charges=item['charges'],
                articles=item['articles'],
                judgment='',
                is_positive=item['is_positive'],
            ))
        
        embeddings = np.load(index_path)
        return cases, embeddings
    
    def _init_strategy(self):
        """初始化精筛策略"""
        config = self.config
        if config.adaptive_mode == "static":
            if config.static_config is None:
                config.static_config = StaticAdaptiveKConfig(
                    min_k=config.min_k,
                    max_k=config.max_k,
                )
            self.strategy: AdaptiveKStrategy = StaticAdaptiveKStrategy(config.static_config)
        elif config.adaptive_mode == "llm":
            if config.llm_config is None:
                config.llm_config = LLMVerifiedAdaptiveKConfig(
                    min_k=config.min_k,
                    max_k=config.max_k,
                )
            self.strategy: AdaptiveKStrategy = LLMVerifiedAdaptiveKStrategy(
                config.llm_config,
                self.llm_client,
                self.llm_model,
            )
        else:
            raise ValueError(f"Unknown adaptive_mode: {config.adaptive_mode}")
    
    def retrieve_topk(
        self,
        target_embedding: np.ndarray,
        target_fact: str,
        candidate_charges: List[str],
        k: Optional[int] = None,
    ) -> Tuple[List[Case], float, int]:
        """分层检索top-k
        Args:
            target_embedding: 目标案件embedding
            target_fact: 目标案件事实（给LLM验证用）
            candidate_charges: 大模型预测的候选罪名
            k: 固定k，None则自适应
        Returns:
            (top_cases, max_sim, final_k)
        """
        logger.info(f"[分层检索] 第一层：根据候选罪名过滤范围，候选罪名={candidate_charges}")
        
        # 收集所有候选罪名对应的案例和embedding
        all_cases: List[Case] = []
        all_embeddings: List[np.ndarray] = []
        
        for charge in candidate_charges:
            # 格式化罪名（去掉罪后缀），并且替换文件名特殊字符
            c = charge.strip()
            if c.endswith("罪"):
                c = c[:-1]
            # 保存文件名时 / 被替换成 _，查找时也替换
            c_safe = c.replace('/', '_')
            # 从本分组找，先试安全名，找不到再试原名
            if c_safe in self.pos_cases:
                logger.info(f"[分层检索] 找到罪名 '{c}'，包含 {len(self.pos_cases[c_safe])} 个候选案例")
                all_cases.extend(self.pos_cases[c_safe])
                all_embeddings.append(self.pos_embeddings[c_safe])
            elif c in self.pos_cases:
                logger.info(f"[分层检索] 找到罪名 '{c}'，包含 {len(self.pos_cases[c])} 个候选案例")
                all_cases.extend(self.pos_cases[c])
                all_embeddings.append(self.pos_embeddings[c])
            else:
                logger.warning(f"[分层检索] 候选罪名 '{c}' 在索引中不存在，跳过")
        
        if not all_cases:
            # 如果找不到， fallback 到所有罪名
            logger.warning(f"[分层检索] 未找到匹配候选案例，回退到全量索引")
            for charge in self.pos_charge_list:
                if charge in self.pos_cases:
                    all_cases.extend(self.pos_cases[charge])
                    all_embeddings.append(self.pos_embeddings[charge])
        
        logger.info(f"[分层检索] 第一层过滤完成，剩余 {len(all_cases)} 个候选案例")
        
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
        coarse_cases = [all_cases[i] for i in top_indices]
        logger.info(f"[分层检索] 第二层：embedding粗筛，选出top-{coarse_k}候选")
        
        # 精筛：用自适应策略计算最终k
        sorted_coarse = [(similarities[i], coarse_cases[i]) for i in top_indices]
        sorted_coarse.sort(key=lambda x: -x[0])
        sorted_similarities = np.array([x[0] for x in sorted_coarse])
        sorted_cases = [x[1] for x in sorted_coarse]
        
        logger.info(f"[分层检索] 第三层：自适应精筛，计算最终k")
        final_k = self.strategy.calculate_k(sorted_similarities, sorted_cases, target_fact)
        final_cases = sorted_cases[:final_k]
        
        logger.info(f"[分层检索] 完成，最终选出 {len(final_cases)} 个案例，最大相似度={max_sim:.4f}")
        
        return final_cases, max_sim, final_k


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
        # 收集所有正例embedding
        pos_embs: List[np.ndarray] = []
        for charge in candidate_charges:
            c = charge.strip()
            if c.endswith("罪"):
                c = c[:-1]
            if c in self.pos_retriever.pos_embeddings:
                pos_embs.append(self.pos_retriever.pos_embeddings[c])
        if pos_embs:
            all_pos_embs = np.concatenate(pos_embs, axis=0)
            distances = [1 - (embed @ target_norm) for embed in all_pos_embs[positive_examples]]
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


def create_hierarchical_retriever_from_config(
    config_dict: dict,
    llm_client: Optional[Any] = None,
    llm_model: Optional[str] = None,
) -> HierarchicalAdaptiveRAGRetriever:
    """从配置创建分层检索器"""
    
    def _parse_embedding_config(cfg: dict) -> HierarchicalRetrieverConfig:
        adaptive_mode = cfg.get("adaptive_mode", "llm")
        
        static_config = None
        if adaptive_mode == "static":
            static_kwargs = cfg.get("static", {})
            static_config = StaticAdaptiveKConfig(**static_kwargs)
        
        llm_config = None
        if adaptive_mode == "llm":
            llm_kwargs = cfg.get("llm", {})
            llm_config = LLMVerifiedAdaptiveKConfig(**llm_kwargs)
        
        return HierarchicalRetrieverConfig(
            index_root=cfg.get("index_root", "data/index_by_charge"),
            embedding_model_name=cfg.get("embedding_model_name", "uer/sbert-base-chinese-nli"),
            min_k=cfg.get("min_k", 1),
            max_k=cfg.get("max_k", 5),
            coarse_k=cfg.get("coarse_k", 10),
            adaptive_mode=adaptive_mode,
            static_config=static_config,
            llm_config=llm_config,
        )
    
    pos_dict = config_dict.get("positive", {})
    neg_dict = config_dict.get("negative", {})
    
    pos_config = _parse_embedding_config(pos_dict)
    neg_config = _parse_embedding_config(neg_dict)
    
    pos_retriever = HierarchicalEmbeddingRetriever(pos_config, llm_client, llm_model)
    neg_retriever = HierarchicalEmbeddingRetriever(neg_config, llm_client, llm_model)
    
    return HierarchicalAdaptiveRAGRetriever(pos_retriever, neg_retriever)


if __name__ == "__main__":
    # 简单测试
    print("EmbeddingRetriever 模块加载成功，支持两种自适应k策略：")
    print("  - static: 静态数学公式")
    print("  - llm: 大模型验证打分")
    print("  - hierarchical: 分层检索（罪名过滤 → 粗筛 → 精筛）")

