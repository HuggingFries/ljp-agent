# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

LJP-RAG Agent: Legal Judgment Prediction on Chinese criminal cases (CAIL2018) using Retrieval-Augmented Generation. The agent extracts 7 legal elements from a case fact, retrieves similar historical cases from a unified knowledge base via embedding similarity, then uses an LLM (DeepSeek) to predict charges, articles, sentence term (prison months), and fine.

**Key design:** LLM outputs free-text charge names (no 202-item charge list in prompt). A `ChargeMatcher` maps outputs to standard CAIL2018 names via Sentence-BERT cosine similarity.

## Project Structure

- `src/agent/` — RAG pipeline (module, import as `from src.agent.agent import LJPRAGAgent`)
  - `element_extractor.py` — LLM extracts 7 legal elements from a case fact
  - `retriever.py` — Sentence-BERT weighted embedding search against unified index
  - `charge_matcher.py` — Embedding-based mapper: free-text charge names → standard CAIL2018 names
  - `article_matcher.py` — Article validation with charge→article mapping (from `data/charge_article_mapping.json`); first correction shows charge-intersection articles, subsequent attempts show full list
  - `agent.py` — Orchestrator: extract → retrieve → format prompt → LLM predict (3 tasks); article feedback loop with charge-aware narrowing
- `src/baseline/baseline.py` — Pure LLM baseline (no RAG), same predict() interface
- `scripts/` — KB construction pipeline scripts (run individually, not a module)
  - `collect_negative_kb.py` — Collect error cases (stratified by type: charge/article/term) with inline quality validation
  - `build_hierarchical_error.py` — Build L1 (elements) + L2 (rule/reasoning/error_reason) layers
  - `build_hierarchical_index.py` — Embed L1 and build index
- `config/` — YAML configs (`config.yaml` for agent, `kb_building.yaml` for KB construction)
- `data/` — CAIL2018 dataset, accusation/law lists, pre-built indices, KB JSON files, charge→article mapping (`charge_article_mapping.json`)
- `test_with_wandb.py` — Evaluation harness with wandb logging, concurrent execution

## Config & API Setup

Config lives in `config/config.yaml`. API config is read from env vars:
- `DEEPSEEK_API_KEY` (or `OPENAI_API_KEY`)
- The model in config is `deepseek-v4-flash` at `https://api.deepseek.com/v1`

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test with a single case
python src/agent/agent.py --fact data/sample.txt

# Run evaluation with wandb logging
python test_with_wandb.py --max-samples 50 --run-all

# Test retrieval quality on random samples
python test_retrieval.py --top-k 5 --num-test 3

# Full KB pipeline (in order):
python scripts/collect_negative_kb.py         # 1. Collect error cases (stratified, with inline cleaning)
python scripts/build_hierarchical_error.py  # 2. Build L1+L2 analysis
python scripts/build_hierarchical_index.py     # 3. Build embedding index
```

## KB Data Layers

Each case in the hierarchical KB has 4 layers:
- **L0**: Raw original data (fact, predictions, true labels, term, fine)
- **L1**: 7 legal elements extracted from fact (used for embedding retrieval)
- **L2**: Cognitive layer — case_summary + rule (legal大前提) + reasoning (涵摄) + error_reason (易错点分析)
- **L3**: Reserved

The retriever only embeds L1.legal_elements. Query embedding is a weighted combination (config: elements_weight=0.7, fact_weight=0.3) of target elements and original fact.

## KB Construction Pipeline

Two-stage pipeline:

1. **Collection** (`collect_negative_kb.py`): LLM predicts charges + articles + term on training set. Stratified error collection: per charge, collects n cases of each error type (charge_error / article_error / term_error) = 3n total per charge (n=1 by default, ~606 total). Inline quality validation discards empty/short reasoning, empty charges/articles, invalid term. Smart filtering + worker-pull loop removes completed charges from processing pool.

2. **Hierarchical build** (`build_hierarchical_error.py`): For each case, generates L1 (7 legal elements for retrieval) and L2 (case_summary + rule + reasoning + error_reason).

3. **Indexing** (`build_hierarchical_index.py`): Embeds L1 layer, outputs unified_* files.

## Evaluation Metrics

- **Charge accuracy**: Exact set match between predicted and true charges
- **Article accuracy**: Exact set match for articles (only for samples with article data)
- **Joint accuracy**: Charge + article + term all correct
- **Term accuracy**: Death/life boolean exact match; imprisonment within tolerance (max(20%, 12 months))
- **Term MAE**: Mean absolute error in months (fixed-term sentences only)
- **Fine accuracy**: Fine within tolerance (max(20%, 1000 RMB))
- **Fine MAE**: Mean absolute error in RMB

## Project Conventions

- **Comments**: English only, short, explain what code does
- **Naming**: Keep result/log filenames minimal
- **API calls**: Default to parallel (ThreadPoolExecutor), never serial
- **Config**: Hyperparameters in config YAML, scripts accept `--config` overrides
- **Dependencies**: Add all new deps to `requirements.txt`
- **API keys**: Prefer `DEEPSEEK_API_KEY` env var; fallback `OPENAI_API_KEY`
- **Code separation**: Different pipeline stages are separate scripts, don't import each other
- **Test set isolation**: Never use test set for distillation or KB construction
