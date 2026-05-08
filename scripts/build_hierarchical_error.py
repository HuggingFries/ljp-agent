#!/usr/bin/env python3
"""
Build hierarchical knowledge base: L1 legal elements + L2 cognitive layer (rule/reasoning/error_reason).
Data collection done separately by collect_negative_kb.py.

Usage:
  python scripts/build_hierarchical_error.py [options]
Options:
  --config CONFIG       Config file path (default: config/kb_building.yaml)
  --input PATH          Input collected raw errors
  --output PATH         Output hierarchical json
  --max-workers N       Parallel API workers (default: 10)
  --max-samples N       Only process first N samples (for testing)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
# Ensure project root is in sys.path so that src/ can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

# --------------------------- Prompt Templates ---------------------------

ELEMENT_EXTRACTION_PROMPT = """你是一位法律AI助手，请你帮我从以下刑事案件中提取七个**定性法律要素**。

### 提取要求：
- **只提取性质判断，不要提取具体人名、地名、具体数值等个性化信息**
- 只保留和定罪相关的法律性质，过滤噪声信息
- 帮助我们检索"在法律要素上相似"的错例，不是检索事实细节相似

请严格按照JSON格式输出，包含以下七个字段：
1. 犯罪主体：单人/多人/单位；身份特点（如国家工作人员/普通公民）；是否有前科等（只写性质，不写姓名）
2. 犯罪行为：行为的性质类型（如：秘密窃取/暴力胁迫/欺诈骗取等）
3. 犯罪手段：实施行为的手段特点（如：持刀/投放危险物质/利用信息网络等）
4. 犯罪客体：行为侵犯的客体类别（如：公共安全/公民人身权利/财产权利等）
5. 犯罪动机：主观罪过形式（故意/过失；犯罪动机是什么）
6. 危害程度：危害结果的严重程度（造成死亡/造成轻伤/数额较大/巨大等）
7. 法益类型：具体侵犯的法益类型（如：盗窃罪侵犯财产所有权；故意伤害侵犯身体权）

案件事实如下：

{fact}

仅输出JSON，不要输出任何其他内容。"""

L2_GENERATION_PROMPT = """你是一位法律认知分析专家。请分析以下刑事案件，生成结构化法律分析。

**案件信息：**
真实罪名：{true_charges_text}
相关法条：{true_articles_text}
刑期：{true_term_text}

**法律要素（案件关键特征的抽象）**：
{l1_elements_json}

**案件事实：**
{fact}

**错误类型说明：**
{error_context}

---

请按以下四步分析，输出严格JSON格式。注意：rule和reasoning必须覆盖**罪名和刑期**两个维度。

1. **case_summary（事实摘要）**
   - 1-2句话概括核心事实：行为性质、手段、后果或数额
   - 只保留定罪量刑必需的要素，省略细节

2. **rule（判案规则——大前提）**
   - {rule_intro}
   - **只写抽象构成要件**，不写法条编号、不写具体数值、不涉及本案事实
   - 2-3句，不写典型场景
   - **只描述本条规则本身，不要与其他选项做对比**

3. **reasoning（涵摄分析——小前提）**
   - 将rule的抽象要件与本案具体事实一一匹配
   - 格式：[事实中的关键情节]→[满足哪一要件]→[结论]
   - 3-5句，**只写fact中确实存在的事实，不要编造或推断未提及的细节**

4. **error_reason（易错点分析）**
   - {error_reason_intro}
   - 1-2句话，只针对本案例实际发生的错误类型展开

输出格式：
{{
  "case_summary": "...",
  "rule": "...",
  "reasoning": "...",
  "error_reason": "..."
}}

仅输出JSON，不要任何其他内容。"""


# --------------------------- Helper Functions ---------------------------

def run_l1_extraction(client: OpenAI, model: str, fact: str) -> Optional[Dict[str, str]]:
    """Extract 7 legal elements (L1) from case fact."""
    prompt = ELEMENT_EXTRACTION_PROMPT.format(fact=fact)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        content = content.removeprefix("```json").removesuffix("```").strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"L1 extraction failed: {e}")
        return None


def run_l2_generation(client: OpenAI, model: str, fact: str, l1_elements: Dict,
                      true_charges: List[str], true_articles: List[str],
                      true_term: Dict = None,
                      pred_charges: List[str] = None, pred_articles: List[str] = None,
                      pred_term: Dict = None) -> Optional[Dict]:
    """Generate L2: case_summary, rule, reasoning, error_reason.
    Dynamically focuses on the actual error type (charge/article/term)."""
    true_charges_text = "、".join(true_charges) if true_charges else "无"
    true_articles_text = "、".join(true_articles) if true_articles else "无"
    true_term_text = _format_term(true_term) if true_term else "无"
    l1_elements_json = json.dumps(l1_elements, ensure_ascii=False, indent=2)

    # Determine primary error type (charge > article > term priority)
    is_charge_error = pred_charges and set(pred_charges) != set(true_charges)
    is_article_error = pred_articles and set(pred_articles) != set(true_articles)
    is_term_error = pred_term and not is_term_accurate(true_term, pred_term)

    if is_charge_error:
        false_label = "、".join(pred_charges)
        error_context = f"本案例罪名预测错误：正确应为{true_charges_text}，但被误判为{false_label}。"
        rule_intro = f"抽象描述{true_charges_text}的罪名构成要件及量刑因素"
        error_reason_intro = f"本案例真实为{true_charges_text}但曾被误判为{false_label}。请分析{true_charges_text}与{false_label}之间的关键区分点，正面说明两者的本质区别，帮助模型在不同案件中准确区分"
    elif is_article_error:
        false_label = "第" + "、".join(pred_articles) + "条"
        true_art_label = "第" + "、".join(true_articles) + "条"
        error_context = f"本案例罪名正确（{true_charges_text}）但法条错误：正确为{true_art_label}，被误判为{false_label}。"
        rule_intro = f"抽象描述{true_charges_text}的罪名构成要件及量刑因素"
        error_reason_intro = f"本案例正确法条为{true_art_label}但曾被误判为{false_label}。请分析同一罪名下不同法条容易混淆的原因，说明哪些情节差异决定法条选择"
    elif is_term_error:
        error_context = f"本案例罪名法条均正确但刑期预测存在偏差。"
        rule_intro = f"抽象描述{true_charges_text}的罪名构成要件及量刑因素"
        error_reason_intro = f"本案例刑期预测存在偏差。请分析哪些量刑因素容易忽视或误判，说明该类案件的量刑要点"
    else:
        error_context = f"本案例各项预测均正确。"
        rule_intro = f"抽象描述{true_charges_text}的罪名构成要件及量刑因素"
        error_reason_intro = "本案例各项预测均正确，无需易错点分析"

    prompt = L2_GENERATION_PROMPT.format(
        fact=fact,
        true_charges_text=true_charges_text,
        true_articles_text=true_articles_text,
        true_term_text=true_term_text,
        l1_elements_json=l1_elements_json,
        error_context=error_context,
        rule_intro=rule_intro,
        error_reason_intro=error_reason_intro,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        content = content.removeprefix("```json").removesuffix("```").strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"L2 generation failed: {e}")
        return None


def _format_term(term: dict) -> str:
    """Format term dict into readable text."""
    if term.get("death_penalty"):
        return "死刑"
    if term.get("life_imprisonment"):
        return "无期徒刑"
    imp = term.get("imprisonment", 0)
    if imp > 0:
        years = imp / 12
        return f"有期徒刑{imp}个月（{years:.1f}年）"
    return "无刑期"


def is_term_accurate(true_term: dict, pred_term: dict) -> bool:
    """Check if predicted term matches true within tolerance."""
    if true_term.get("death_penalty") != pred_term.get("death_penalty"):
        return False
    if true_term.get("life_imprisonment") != pred_term.get("life_imprisonment"):
        return False
    true_imp = true_term.get("imprisonment", 0)
    pred_imp = pred_term.get("imprisonment", 0)
    tolerance = max(true_imp * 0.2, 12)
    return abs(true_imp - pred_imp) <= tolerance


def process_case(client: OpenAI, model: str, case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fact = case.get("fact", "")
    pred_reasoning = case.get("pred_reasoning", "")
    predicted_charges = case.get("predicted_charges", [])
    true_charges = case.get("true_charges", [])
    true_articles = case.get("true_articles", [])
    predicted_articles = case.get("predicted_articles", [])
    true_term = case.get("true_term", {})
    pred_term = case.get("predicted_term", {})
    true_fine = case.get("true_fine", 0)
    predicted_fine = case.get("predicted_fine", 0)

    if not fact or not pred_reasoning:
        logger.warning("Missing fact or reasoning, skipping")
        return None

    # 1. L1 extraction (retrieval layer)
    l1_elements = run_l1_extraction(client, model, fact)
    if not l1_elements:
        return None

    # 2. L2 generation (rule + reasoning + error_reason)
    analysis = run_l2_generation(client, model, fact, l1_elements,
                                 true_charges, true_articles,
                                 true_term, predicted_charges, predicted_articles, pred_term)
    if not analysis:
        return None

    reasoning = analysis.get("reasoning", "")
    error_reason = analysis.get("error_reason", "")

    # 3. Build output with L0+L1+L2+L3
    L0 = {
        "fact": fact,
        "true_charges": true_charges,
        "predicted_charges": predicted_charges,
        "true_articles": true_articles,
        "predicted_articles": predicted_articles,
        "true_term": true_term,
        "predicted_term": pred_term,
        "true_fine": true_fine,
        "predicted_fine": predicted_fine,
        "pred_reasoning": pred_reasoning,
        "predict_prompt_tokens": case.get("predict_prompt_tokens", 0),
        "predict_completion_tokens": case.get("predict_completion_tokens", 0),
    }
    L1 = {"legal_elements": l1_elements}
    L2 = {
        "case_summary": analysis.get("case_summary", ""),
        "rule": analysis.get("rule", ""),
        "reasoning": reasoning,
        "error_reason": error_reason,
    }
    L3 = {}

    return {
        "L0": L0,
        "L1": L1,
        "L2": L2,
        "L3": L3,
    }


# --------------------------- Main Pipeline ---------------------------

def load_api_config(global_config: dict):
    api_config = global_config.get("api", {})
    base_url = api_config.get("base_url")
    api_key_env_var = api_config.get("api_key", "DEEPSEEK_API_KEY")
    model_name = api_config.get("model_name")

    api_key = os.getenv(api_key_env_var)
    if not all([base_url, api_key, model_name]):
        raise ValueError(f"Missing API config: base_url, {api_key_env_var}, model_name.")
    return base_url, api_key, model_name


def main():
    parser = argparse.ArgumentParser(description='Build hierarchical KB: L1 legal elements + L2 cognitive layer (rule/reasoning/error_reason)')
    parser.add_argument('--config', type=str, default=str(ROOT_DIR / 'config/kb_building.yaml'))
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()

    # Configs
    with open(args.config, 'r', encoding='utf-8') as f:
        kb_config = yaml.safe_load(f)
    with open(ROOT_DIR / 'config/config.yaml', 'r', encoding='utf-8') as f:
        global_config = yaml.safe_load(f)

    build_config = kb_config.get("hierarchical_build", {})
    input_path = args.input or build_config.get("input", str(ROOT_DIR / "data/negative_error_cases/collected_errors.json"))
    output_path = args.output or build_config.get("output", str(ROOT_DIR / "data/negative_error_cases/collected_errors_hierarchical.json"))
    max_workers = args.max_workers or build_config.get("max_workers", 10)

    base_url, api_key, model_name = load_api_config(global_config)
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Load input
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    if isinstance(input_data, dict) and "error_cases" in input_data:
        cases = input_data["error_cases"]
        metadata = input_data.get("metadata", {})
    else:
        logger.error("Invalid input format")
        sys.exit(1)
    logger.info(f"Loaded {len(cases)} raw error cases")

    indexed_cases = list(enumerate(cases))
    if args.max_samples:
        indexed_cases = indexed_cases[:args.max_samples]
        logger.info(f"Testing mode: limited to {len(indexed_cases)} samples")

    if not indexed_cases:
        logger.info("No cases to process.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pbar = tqdm(total=len(indexed_cases)) if has_tqdm else None

    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for orig_idx, case in indexed_cases:
            future = executor.submit(process_case, client, model_name, case)
            futures[future] = orig_idx

        for future in as_completed(futures):
            orig_idx = futures[future]
            result = future.result()
            if result is None:
                if pbar:
                    pbar.update(1)
                continue

            result["original_index"] = orig_idx
            all_results.append(result)

            if pbar:
                pbar.update(1)

            if len(all_results) % 10 == 0:
                current_metadata = dict(metadata)
                current_metadata["total_cases"] = len(all_results)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({"metadata": current_metadata, "cases": all_results}, f, ensure_ascii=False, indent=2)

    if pbar:
        pbar.close()

    final_metadata = dict(metadata)
    final_metadata["total_cases"] = len(all_results)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"metadata": final_metadata, "cases": all_results}, f, ensure_ascii=False, indent=2)

    logger.info(f"Done! Saved {len(all_results)} cases to {output_path}")


if __name__ == "__main__":
    main()
