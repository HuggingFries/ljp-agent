# LJP-RAG: 基于正负案例对比RAG的法律判决预测

基于RAG+prompt工程的法律判决预测智能体，核心思想是引入**正负案例对比**作为上下文，提升大语言模型判决预测准确性。

v1.1新增**自适应k检索**：根据输入案件与检索库的最大相似度，动态调整检索案例数。相似度越高，k越小；相似度越低，k越大。

## 项目结构

```
ljp-agent/
├── agent.py                 # 核心RAG智能体框架
├── retriever.py             # 检索器：支持固定k和自适应k
├── main.py                  # 主入口，API配置加载
├── run_agent.py             # 独立运行脚本：批量评估 + 单案件预测
├── compare_baseline_rag.py  # 基线模型与RAG对比实验
├── baseline.py              # Zero-Shot基线实现
├── analyze_sim_dist.py      # 相似度分布分析工具
├── requirements.txt         # Python依赖
└── data/
    ├── cail2018/            # CAIL2018数据集（标签文件）
    └── ...                 # 生成的正负案例索引
```

## 核心方法

1. **正例检索**: 检索与目标案件最相似的已判决案例作为正例，告诉模型"类似案件应该怎么判"
2. **负例选择**: 选择易混淆/不相似案例作为负例，告诉模型"不要怎么判"
3. **自适应k**: 根据最大相似度动态调整检索案例数，避免冗余也保证信息充分
4. **prompt工程**: 将正负案例组织成prompt，输入大模型进行判决预测

## 环境配置

```bash
# 创建conda环境
conda create -n ljp-agent python=3.11
conda activate ljp-agent

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 配置API密钥
cp config.json.example config.json
# 编辑config.json填入你的API信息
{
  "base_url": "https://your-api-base-url.com/v1",
  "api_key": "your-api-key",
  "model_name": "your-model-name"
}
```

## 数据准备

本项目使用CAIL2018数据集：

1. 从[CAIL2018](https://github.com/china-ai-law-challenge/CAIL2018)下载数据
2. 放入`data/cail2018/`目录
3. 运行数据处理脚本分割正负案例（或使用预处理好的索引）

## 使用方法

### 1. 批量评估（测试集评估准确率）

```bash
# 自适应k（推荐，min_k=1, max_k=5）
python run_agent.py --max-samples 100 --min-k 1 --max-k 5

# 固定k对比
python run_agent.py --max-samples 100 --k-positive 3 --k-negative 2

# 开启相似度归一化，让k分布更均匀
python run_agent.py --max-samples 100 --min-k 1 --max-k 5 --normalize

# 指定自定义测试文件
python run_agent.py --test-file /path/to/test.json --max-samples 100

# 结果会保存在 results/ 目录，包含：
# - 整体准确率、precision、recall、F1
# - 每个样本的预测详情和自适应k信息
```

### 2. 单案件预测（直接输入事实，输出判决）

```bash
# 直接输入案件事实
python run_agent.py --fact "被告人张三于2023年1月1日，在北京市朝阳区盗窃他人人民币五千元，数额较大"

# 从JSON文件输入（需包含fact字段）
python run_agent.py --input case.json --output result.json

# 输出会包含预测罪名、相关法条、完整判决理由，以及自适应检索信息
```

### 3. 与Zero-Shot基线对比

```bash
# 对比基线和RAG性能
python compare_baseline_rag.py --max-samples 100
```

## 自适应k算法

```python
# 公式：k = round(min_k + (max_k - min_k) * (1 - sim_max))
# - sim_max越高（找到高相似案例）→ k越小，结果聚焦
# - sim_max越低（没找到高相似案例）→ k越大，提供更多参考

# 例子：
sim_max=0.9 → k≈1  (高相似，只需要最相似的1个案例)
sim_max=0.5 → k≈3  (中等相似，需要3个参考)
sim_max=0.1 → k≈5  (低相似，需要最多5个参考)
```

## 实验结果

在CAIL2018 100随机样本上：

| 方法 | 罪名准确率 | 法条准确率 |
|------|-----------|-----------|
| Zero-Shot基线 | XX.XX% | XX.XX% |
| RAG + 固定k (pos=3, neg=2) | +XX% | +XX% |
| RAG + 自适应k (1-5) | **XX.XX%** | **XX.XX%** |

*待补充完整实验结果*

## 负例选择策略

- **farthest**: 选择距离最远（最不相似）作为负例
- **random**: 随机从剩余案例中选择

## 数据集

- [CAIL2018](https://github.com/china-ai-law-challenge/CAIL2018): 中国法律智能挑战数据集，标准LJP数据集

## License

MIT
