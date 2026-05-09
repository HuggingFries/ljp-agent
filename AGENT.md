# Agent 工作须知

## 一、代码规范

### 1. 注释规范
- 使用英文注释，保持简洁简短。
- 注释只解释“代码在做什么”，不要包含实验方向、修改记录、用户请求、警示、禁止事项等冗余信息。
- 文件头部注释：仅包含本文件的作用，以及附带参数的用法（如有）。
- 注释的目的是帮助人类阅读者理解代码，不是给Agent写日记。

### 2. 结果与日志命名
- 命名保持简洁，只保留最基本的身份标识。
- 例如：要求“并行调用API”时，不要在结果或日志文件名中加入“parallel”字样。用 `results_n100.csv` 或 `log_modelA.txt` 即可。

### 3. 代码执行方式
- 默认写成并行调用API，不要使用单线程串行方式。

### 4. 超参数配置
- 默认将超参数放在config文件中，不要直接写死在脚本里。
- 脚本需支持运行时修改参数（例如通过命令行参数 `--config` 或 `--lr` 等）。
- config不要过于嵌套：不同功能的脚本使用独立的config文件，并分好类。尽量避免在每个类别里再分小类别。

### 5. 依赖管理
- 所有新增依赖请写入 `requirements.txt`

### 6. API使用规范
- 默认使用 **DeepSeek API**，兼容OpenAI接口格式
- LLM调用统一走OpenAI兼容模式
- API密钥读取逻辑：优先读`OPENAI_API_KEY`，回退到`DEEPSEEK_API_KEY`环境变量
- 本项目使用：`DEEPSEEK_API_KEY`

### 5. 代码结构与解耦
- 代码要接口化、功能分离。
- 不要出现不同功能的脚本相互调用（例如：训练脚本不要直接调用数据预处理脚本的函数）。

### 6. 实验/探索性脚本的位置
- 与主项目运行无关的脚本（如实验、探索性质的代码）不要放在根目录下，请统一放在 `src/` 目录中。

## 二、实验规范

### 1. 知识库构建
- 不要使用测试集进行蒸馏。这会导致数据泄露和跑分虚高。

### 2. 动态更新项目框架
- 每次写完代码后，请在本须知的末尾更新“当前项目框架”部分。
- 内容包括：项目工作流、脚本组织、实验方向等。
- 若实验方向发生变化，也请同步修改。

### 3. 删除脚本或弃用idea
- 删除所有配套部分，不留残留。
- 优先参考本须知末尾动态更新的项目框架来定位相关文件；若搞不明白，再遍历项目文件彻底删除。
- 删除后务必更新本须知末尾的框架简介。

---

# 当前项目框架（动态更新）

> Agent 每次修改代码后，请在此处更新以下内容：
> - 项目工作流（数据加载 → 预处理 → 训练 → 评估 等）
> - 主要脚本及其作用
> - 当前实验方向或活跃的idea

## 项目：LJP-RAG 统一历史案例增强

### 项目目标
构建基于**统一历史案例知识库**的RAG系统，每个案例包含：
- **正例**（正确判决案例）：从训练集选取的典型罪名正确判决
- **错例**（错误分析案例）：LLM曾预测错误的案例，含规则（rule）、涵摄（reasoning）、易错点（error_reason）三维分析
帮助LLM参考历史案例进行三任务联合预测（罪名+法条+刑期+罚金）。

### 完整工作流

知识库构建流程：

1. **错误收集（`collect_negative_kb.py`）**
   - 从CAIL2018训练集用纯LLM预测，识别预测错误的案例
   - LLM prompt中不提供202罪名列表；输出的自由文本罪名通过`ChargeMatcher`嵌入映射到标准名称，法条通过`ArticleMatcher`做修正循环
   - 每个罪名收集3种错误各n条（charge_error / article_error / term_error），满员自动剪枝
   - 质量过滤：reasoning<50字、空罪名/空法条、无效刑期均丢弃
   - 支持断点续传（`--resume-from`）

2. **分层构建（`build_hierarchical_error.py`）**
   - **L1（检索层）**：提取7个定性法律要素（犯罪主体/行为/手段/客体/动机/危害程度/法益类型），用于嵌入检索
   - **L2（认知层）**：四段式分析
     - `case_summary`：事实摘要（保留定罪量刑关键信息）
     - `rule`：判案规则（大前提，抽象法律构成要件，不绑定本案细节）
     - `reasoning`：涵摄分析（小前提，三段论：rule→本案事实→结论）
     - `error_reason`：易错点分析（结合模型实际错误，分析哪些要素导致误判）
   - 根据实际错误类型（charge > article > term）动态调整rule/error_reason的生成焦点

3. **索引构建（`build_hierarchical_index.py`）**
   - 对L1层进行Sentence-BERT嵌入，归一化后输出`unified_*`文件
   - 输出位置：`data/index_hierarchical/unified_*`
   - 注意：当前目录只有`pos_*`/`neg_*`，需运行此脚本生成unified索引

5. **检索阶段**
   - 对输入案件提取七要素，加权组合（elements_weight=0.7, fact_weight=0.3）进行余弦相似度检索
   - 从统一知识库中检索top-k最相似的历史案例

6. **预测阶段**
   - 基线模式（baseline）：纯LLM预测，不注入历史案例
   - Agent模式（RAG增强）：检索到的案例注入prompt，展示case_summary + rule + reasoning + error_reason
   - 两模式均通过`ChargeMatcher`映射罪名
   - Agent模式额外使用`ArticleMatcher`做法条合法性修正循环（带罪名关联法条交集筛选：首轮显示罪名交集法条，后续回退全量列表）

7. **评估阶段**
   - 罪名准确率：集合精确匹配
   - 法条准确率：集合精确匹配
   - 联合准确率：罪名+法条+刑期三者同时正确
   - 刑期准确率：死刑/无期布尔精确匹配；有期徒刑容忍度 max(20%, 12月)
   - 刑期MAE：有期徒刑MAE（月）
   - 罚金准确率：容忍度 max(20%, 1000元)
   - 罚金MAE：罚金MAE（元）

### KB分层结构

L0: 原始信息层（fact, true/pred charges/articles/term/fine, pred_reasoning）
L1: 检索层（7个定性法律要素，Sentence-BERT嵌入）
L2: 认知层：
   - `case_summary`：事实摘要
   - `rule`：判案规则（抽象大前提）
   - `reasoning`：涵摄分析（三段论）
   - `error_reason`：易错点分析
L3: 当前未使用

### 项目结构

```
test_with_wandb.py       评估脚本，支持baseline/agent，wandb记录
test_retrieval.py        检索质量测试
src/agent/
  agent.py               RAG Agent：extract→retrieve→predict
  element_extractor.py   七要素提取
  retriever.py           统一KB检索器（加权嵌入+余弦相似度）
  charge_matcher.py      罪名映射（Sentence-BERT余弦→标准罪名名）
  article_matcher.py     法条验证+修正循环（带罪名关联法条交集筛选）
src/baseline/
  baseline.py            纯LLM基线
scripts/
  collect_negative_kb.py               错误收集（分层+质量过滤+断点续传）
  build_hierarchical_error.py          L1+L2生成（要素提取+规则/推理/边界）
  build_hierarchical_index.py          嵌入索引构建
  analyze_charge_impact.py             辅助分析
  compare_errors.py                    辅助对比
config/
  config.yaml            Agent配置（api, retriever, index, data）
  kb_building.yaml       KB构建配置（collection, hierarchy_build）
data/
  accu.txt               202个标准罪名
  law.txt                法条编号列表
  charge_article_mapping.json  罪名→法条映射（从训练集预构建）
  index_hierarchical/    索引文件
```

### 项目现状（2026-05-09）
- 负例收集+构建完成：513条分层案例（L1+L2已完成）
- 统一索引已构建：`data/index_hierarchical/unified_*`（513条，768维）
- 三任务联合预测（罪名+法条+刑期+罚金）已实现
- ArticleMatcher法条修正循环已集成，带罪名关联法条交集筛选（首轮交集→第二轮全量）
- Joint准确率定义改为罪名+法条+刑期三者同时正确
- 罪名→法条映射已预构建：`data/charge_article_mapping.json`（202罪名）
- RAG top-2 跑分（500条）：charge=88.0%, article=85.1%, term=86.4%, joint=74.0%
- Baseline 跑分（500条）：charge=86.0%, article=82.8%, term=85.4%, joint=73.0%

---

### 附录：对比式知识库（未来方向）

> 以下内容记录对比式知识库的设计方案，当前未使用。保存以备未来参考。

**动机**：原L2的rule只解释了"为什么不判pred_charge"，缺乏"什么情况下该判pred_charge"的判别性边界，可能导致模型倾向于排斥而非选择。

**方案**：每个charge error（true=A, pred=B）配一个true=B的contrastive case，生成：
- `rule_a`：判A的抽象条件
- `rule_b`：判B的抽象条件
- `case_analysis`：本案为何符合A不符B

**试点状态**：2026-05-06 完成5对混淆对（寻衅滋事↔故意伤害、诈骗↔合同诈骗、抢劫↔盗窃、交通肇事↔过失致人死亡、贪污↔受贿），数据在 `contrastive_kb_pilot.json`。全量190 pair设计就绪，待未来运行。
