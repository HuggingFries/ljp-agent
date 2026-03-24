# LJP正负案例智能体

基于prompt工程的法律判决预测智能体，核心思想是引入**正负案例对比**作为上下文，提升大语言模型判决预测准确性。

## 项目结构

```
ljp-positive-negative-prompt/
├── agent.py           # 核心智能体框架
├── retriever.py       # 正负案例检索器
├── requirements.txt   # Python依赖
└── data/
    └── cail2018/      # CAIL2018数据集
```

## 核心方法

1. **正例检索**: 检索与目标案件最相似的案例作为正例，告诉模型"应该怎么判"
2. **负例选择**: 选择最不相似或易混淆案例作为负例，告诉模型"不应该怎么判"
3. **prompt工程**: 将正负案例组织成prompt，输入大模型进行预测

## 快速开始

```python
from agent import LJPAgent, Case, DataLoader
from retriever import EmbeddingRetriever

# 加载数据
data = DataLoader.load_cail2018("data/cail2018/train.json")
cases = [DataLoader.convert_to_case(item) for item in data]

# 构建索引
retriever = EmbeddingRetriever(embedding_model=your_embedding_model)
retriever.index(cases)

# 检索正负案例
result = retriever.retrieve(target_embedding, k_positive=5, k_negative=2)

# 构建prompt预测
agent = LJPAgent()
prompt = agent.build_prompt(target_case, result.positive_examples, result.negative_examples)
```

## 负例选择策略

- **farthest**: 选择距离最远（最不相似）作为负例
- **random**: 随机从剩余案例中选择

## 数据集

- [CAIL2018](https://github.com/china-ai-law-challenge/CAIL2018): 中国法律智能挑战数据集，标准LJP数据集
