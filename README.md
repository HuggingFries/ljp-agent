# LJP-RAG Agent

Legal Judgment Prediction with Retrieval-Augmented Generation and Adaptive k.

支持多种自适应k策略，自动决定检索多少个案例。

## 配置

所有超参数都在 `config.json` 中配置：

```json
{
  "api": {
    "base_url_env": "OPENAI_BASE_URL",    // 从环境变量读API base url
    "api_key_env": "OPENAI_API_KEY",      // 从环境变量读API key
    "model_name_env": "OPENAI_MODEL"      // 从环境变量读模型名称
  },
  "retriever": {
    "positive": {
      "adaptive_mode": "static",         // 自适应k模式: "static" | "llm"
      "static": {                       // static 模式参数（数学公式计算）
        "min_k": 1,                     // 最小返回k
        "max_k": 5,                     // 最大返回k
        "alpha": 1.0,                   // 缩放系数: k = min_k + (max_k - min_k) * (1 - sim_max) * alpha
        "normalize": false,             // 是否对sim_max做归一化
        "sim_min": null,                // 归一化最小sim（null自动计算）
        "sim_max": null                 // 归一化最大sim（null自动计算）
      },
      "llm": {                         // llm 模式参数（大模型验证相关性）
        "min_k": 1,                     // 最小保证返回
        "max_k": 5,                     // 最大返回，避免上下文溢出
        "candidate_k": 10,              // embedding粗筛给大模型评估的候选数
        "min_score_threshold": 3         // 最低相关性分数（1-5），低于过滤
      }
    },
    "negative": {                       // 负例检索器配置，结构和positive一样
      "adaptive_mode": "static",
      "static": { ... },
      "llm": { ... }
    }
  },
  "index": {
    "index_dir": "data",                // 索引存放目录
    "embedding_model": "uer/sbert-base-chinese-nli"  // embedding模型
  },
  "evaluation": {
    "test_file": "data/final_all_data/first_stage/test.json",  // 默认测试集
    "seed": 42                          // 采样随机种子
  },
  "logging": {
    "level": "INFO"                     // 日志级别
  }
}
```

## 两种自适应k模式

### 1. static 模式（默认）
- **原理**: 根据最大相似度数学公式计算k：`sim_max`越大 → k越小，`sim_max`越小 → k越大
- **优点**: 速度快，不需要额外大模型调用
- **配置**: 在 `config.json` 中设置 `retriever.positive.adaptive_mode = "static"`

### 2. llm 模式（大模型验证）
- **原理**: embedding先粗筛top10，让大模型给每个候选打法律相关性分（1-5），只保留≥阈值的，个数就是k
- **优点**: 解决「语义相似 ≠ 法律相似」问题，大模型懂法律构成要件，过滤掉不相关的
- **配置**: 在 `config.json` 中设置 `retriever.positive.adaptive_mode = "llm"`，negative同理
- **开销**: 每个案件需要 `candidate_k` 次大模型调用（默认10次），精度更高但更慢

正负例可以**独立配置不同模式**，比如：
```json
"positive": { "adaptive_mode": "llm" },
"negative": { "adaptive_mode": "static" }
```

## 运行

### 环境变量
先设置API环境变量（对应config.json里配置的名称）：
```bash
export OPENAI_BASE_URL="your-api-base-url"
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="model-name"
```

### 批量评估
```bash
# 使用默认配置（config.json），采样50个样本
python run_agent.py --max-samples 50

# 指定配置文件，自定义采样数和输出
python run_agent.py --config config-static.json --max-samples 100 --output results/static.json

# 强制固定k
python run_agent.py --max-samples 50 --k-positive 3 --k-negative 1
```

### 单案件预测
```bash
# 直接输入事实
python run_agent.py --fact "被告人张三于2023年1月1日在南京市盗窃了被害人李四的人民币一万元..."

# 从json文件输入
python run_agent.py --input case.json
```

## 文件结构

```
├── config.json          # 主配置文件
├── run_agent.py        # 入口脚本
├── agent.py            # LJP Agent 核心框架
├── retriever.py        # 检索器（支持多种自适应k策略）
├── build_index.py      # 构建embedding索引
├── evaluate_*.py      # 评估脚本
├── requirements.txt   # 依赖
├── data/              # 数据和索引
└── results/          # 评估结果输出
```

## 依赖安装
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 对比实验

方便对比不同自适应k策略：
```bash
# 静态数学公式
cp config.json config-static.json
# 编辑 config-static.json 设置 adaptive_mode: "static"
python run_agent.py --config config-static.json --max-samples 50 --output results/static.json

# 大模型验证
cp config.json config-llm.json
# 编辑 config-llm.json 设置 adaptive_mode: "llm"
python run_agent.py --config config-llm.json --max-samples 50 --output results/llm.json
```

然后比较两个结果文件的准确率就行。
