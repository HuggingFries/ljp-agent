# LJP-RAG Agent

Legal Judgment Prediction on Chinese criminal cases (CAIL2018) using Retrieval-Augmented Generation.

## Overview

The agent extracts 7 legal elements from a case fact, retrieves similar historical cases from a hierarchical knowledge base via embedding similarity, then uses an LLM (DeepSeek) to jointly predict charges, articles, sentence term, and fine.

**Key design:** LLM outputs free-text charge names (no 202-item charge list in prompt). A `ChargeMatcher` maps outputs to standard CAIL2018 names via Sentence-BERT cosine similarity.

## Project Structure

```
├── config/
│   ├── config.yaml           # Agent config (api, retriever, index)
│   └── kb_building.yaml      # KB construction config
├── src/
│   ├── agent/
│   │   ├── agent.py           # RAG orchestrator: extract → retrieve → predict
│   │   ├── element_extractor.py  # 7 legal elements extraction
│   │   ├── retriever.py       # Sentence-BERT weighted embedding search
│   │   ├── charge_matcher.py  # Embedding-based charge name mapping
│   │   └── article_matcher.py # Article validation with charge-aware narrowing
│   ├── baseline/
│   │   └── baseline.py        # Pure LLM baseline (no RAG)
├── scripts/
│   ├── collect_negative_kb.py     # 1. Collect error cases
│   ├── build_hierarchical_error.py# 2. Build L1+L2 analysis
│   └── build_hierarchical_index.py# 3. Build embedding index
├── data/
│   ├── accu.txt               # 202 standard charge names
│   ├── law.txt                # Valid article numbers
│   ├── charge_article_mapping.json  # Pre-built charge→article mapping
│   ├── index_hierarchical/    # Embedding index files
│   └── negative_error_cases/  # KB intermediate files
├── test_with_wandb.py         # Evaluation harness with wandb logging
├── test_retrieval.py          # Retrieval quality test
├── requirements.txt           # Dependencies
├── AGENT.md                   # Agent work guidelines (Chinese)
└── CLAUDE.md                  # Claude Code guidance
```

## KB Data Layers

Each case in the hierarchical KB has 4 layers:

- **L0**: Raw data (fact, predictions, true labels, term, fine)
- **L1**: 7 legal elements (for embedding retrieval)
- **L2**: Cognitive layer — case_summary + rule + reasoning + error_reason
- **L3**: Reserved

## KB Pipeline (in order)

```bash
python scripts/collect_negative_kb.py          # 1. Collect error cases
python scripts/build_hierarchical_error.py     # 2. Build L1+L2 analysis
python scripts/build_hierarchical_index.py     # 3. Build embedding index
```

## Evaluation

```bash
# Baseline (no RAG)
python test_with_wandb.py --max-samples 500 --run-baseline

# RAG agent
python test_with_wandb.py --max-samples 500 --run-agent

# Both
python test_with_wandb.py --max-samples 500 --run-all
```

### Metrics

- **Charge accuracy**: Exact set match
- **Article accuracy**: Exact set match
- **Joint accuracy**: Charge + article + term all correct
- **Term accuracy**: Death/life boolean match; imprisonment within tolerance (max(20%, 12mo))
- **Term MAE**: Mean absolute error in months
- **Fine accuracy/MAE**: Within tolerance (max(20%, 1000 RMB))

## Results (500 samples, top_k=2)

| Metric | Baseline | RAG |
|--------|----------|-----|
| Charge Acc | 86.0% | **88.0%** |
| Article Acc | 82.8% | **85.1%** |
| Term Acc | 85.4% | **86.4%** |
| Joint Acc | 73.0% | **74.0%** |
| Term MAE | 9.3 mo | **9.0 mo** |

## Article Correction

The RAG agent uses a charge-aware narrowing strategy for article correction:
1. LLM predicts freely (no article list in prompt)
2. If articles are invalid → show charge-intersection articles (multi-charge cases use intersection)
3. If still invalid → show full article list

This reduces feedback loop iterations by ~5.7% tokens compared to showing the full list immediately.

## Dependency

```bash
pip install -r requirements.txt
```

## Config

Edit `config/config.yaml` to set:
- API key (via `DEEPSEEK_API_KEY` env var or paste directly)
- Model name
- Retriever `top_k`
- Embedding model
