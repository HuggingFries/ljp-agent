#!/usr/bin/env python3
"""
Multi-model collaborative KB construction.
Layer 1: Data collection (collect_negative_kb.py, done separately).
Layer 2: Build retrieval layer (L1 elements) + cognitive layer (L2 short/long reasoning).
Layer 3: Backtest verification — hide A/C labels, feed B/D reasoning, compute controversy.

Outputs L0/L1/L2/L3 hierarchical structure.

Usage:
  python scripts/build_hierarchical_error_multi.py [options]
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
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
# Ensure project root is in sys.path so that src/ can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from src.agent.charge_matcher import ChargeMatcher

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

L2_GENERATION_PROMPT = """你是一位法律认知分析专家。你将看到一个案件的错误预测信息。请完成两项任务：

**任务一**：对该错误案例进行结构化分析（包含短概括和详细分析）
**任务二**：从该案提炼出1-3条**认知启发**，帮助其他法律AI在遇到类似推理情境时，能更谨慎、更全面地思考。

**背景信息**：
案件事实：
{fact}

模型的错误预测及推理过程：
{pred_reasoning}

正确的判决结果：{true_charges}（法条：{true_articles}）
正确的刑期：{true_term_text}
模型预测的刑期：{pred_term_text}

**任务一要求（结构化分析）**：
请填写以下字段（结合本案具体细节）：

- case_summary：案件关键事实的概括，简洁明了（30字以内）
- correct_reasoning_short：一句话核心概括（20字以内），说明为什么这个判决是正确的
- correct_reasoning_detail：详细的法律三段论分析，包含：
  * 大前提：相关法条规定的构成要件
  * 小前提：本案事实如何符合这些要件
  * 结论：为什么构成该罪及判处该刑期
- wrong_reasoning_short：一句话核心概括（20字以内），说明为什么那个预测判决是错误的
- wrong_reasoning_detail：详细的法律三段论分析，说明为什么预测判决不成立：
  * 大前提：相关法条规定的构成要件
  * 小前提：本案事实缺少哪些要件或存在哪些差异
  * 结论：为什么不构成预测罪名及不应判处该刑期
- error_summary：本案涉及的错误汇总，如果涉及多个错误，分条写出

**任务二要求（认知启发提炼）**：
1. 每条启发只能使用**模糊化、不确定性**的语言。可以使用的句式包括：
   - "或许可以留意..."
   - "建议审视是否存在..."
   - "可能需要检查..."
   - "是否排除了...的可能性？"
   - "在类似情境下，可以考虑..."

2. **绝对禁止**使用以下词汇或句式：
   - "必须"、"应该"、"一定"、"绝对"
   - "禁止"、"不要"、"避免"
   - "正确做法"、"错误做法"
   - 任何形式的强制性指令

3. 启发必须**完全脱离本案的具体细节**（不出现具体罪名、法条、人名、金额等），只能保留抽象的思维方向。

4. 启发不能暗示任何判决结论，也不能暗示哪种思考方向是"正确"的。

5. 即使模型正在审理的案件与当前错误案例完全不同，这些启发也应该是**无害的**。

**示例**（好的启发）：
- "当案件涉及数个相互关联的行为时，或许可以审视这些行为是否被同一概括故意所覆盖。"
- "对于涉及数额门槛的罪名，是否需要确认具体数额是否达到了相应司法解释的标准？"
- "在共同犯罪中，建议审视各参与人的行为是否可能构成不同的罪名。"

**示例**（坏的启发，不要这样写）：
- "必须审查构成要件，不能直接套用熟悉罪名。"
- "禁止忽略对数额门槛的审查。"
- "正确做法是先检查竞合关系。"

输出格式（严格JSON）：
{{
  "case_summary": "案件关键事实概括",
  "correct_reasoning_short": "一句话核心概括：为什么正确",
  "correct_reasoning_detail": "详细三段论：大前提→小前提→结论",
  "wrong_reasoning_short": "一句话核心概括：为什么错误",
  "wrong_reasoning_detail": "详细三段论：大前提→小前提→结论",
  "error_summary": ["错误1", "错误2"],
  "inspirations": ["启发1", "启发2"]
}}

仅输出JSON，不要添加任何其他内容。"""

BACKTEST_WITH_REASONING_PROMPT = """你是一位法律AI。请根据以下案件事实和法律分析意见，独立判断应该判决什么罪名、法条和刑期。

注意：
- 分析意见中的具体罪名和法条已被掩盖为【某罪】/【某条】，请**根据分析逻辑和案件事实**，运用你自己的法律知识做出独立判断。
- 不要直接从分析意见中复制掩盖后的占位符作为输出，应该输出具体的中国刑法罪名名称。
- 法条编号请从可选项中选取，不要编造。

可选法条编号：
{law}

案件事实：
{fact}

分析意见一（支持某判决的理由）：
{correct_reasoning}

分析意见二（反对某判决的理由）：
{wrong_reasoning}

请输出JSON格式：
{{
  "charges": ["罪名1", "罪名2"],
  "articles": ["法条编号"],
  "term": {{"imprisonment": 36, "death_penalty": false, "life_imprisonment": false}},
  "reasoning": "你的推理过程"
}}

仅输出JSON。"""


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


def run_l2_generation(client: OpenAI, model: str, fact: str, pred_reasoning: str,
                      true_charges: List[str], true_articles: List[str],
                      true_term: Dict = None, pred_term: Dict = None) -> Optional[Dict]:
    """Generate L2: case_summary, short+long reasoning, inspirations."""
    true_term_text = _format_term(true_term) if true_term else "无"
    pred_term_text = _format_term(pred_term) if pred_term else "无"
    prompt = L2_GENERATION_PROMPT.format(
        fact=fact,
        pred_reasoning=pred_reasoning,
        true_charges="、".join(true_charges) if true_charges else "无",
        true_articles="、".join(true_articles) if true_articles else "无",
        true_term_text=true_term_text,
        pred_term_text=pred_term_text,
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


def mask_labels_in_text(text: str, charge_names: List[str], article_numbers: List[str],
                        true_term: Dict = None, pred_term: Dict = None) -> str:
    """Replace charge/article/term labels with placeholders for backtest."""
    result = text
    for charge in sorted(charge_names, key=len, reverse=True):
        result = result.replace(charge, "【某罪】")
        cleaned = charge[:-1] if charge.endswith("罪") else charge
        if cleaned != charge:
            result = result.replace(cleaned, "【某罪】")
    for art in sorted(article_numbers, key=len, reverse=True):
        result = result.replace(art, "【某条】")
    # Mask imprisonment values to prevent term answer leakage
    for term_dict in [true_term, pred_term]:
        if not term_dict:
            continue
        imp = term_dict.get("imprisonment", 0)
        if imp and imp > 0:
            result = re.sub(rf'{imp}\s*个?月', '【某月】', result)
            years = round(imp / 12, 1)
            result = re.sub(rf'{years}\s*年', '【某年】', result)
            if imp % 12 == 0:
                year_int = imp // 12
                result = re.sub(rf'(?<!\d){year_int}(?!\d)\s*年', '【某年】', result)
    return result


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


def run_backtest(client: OpenAI, model: str, fact: str,
                 correct_reasoning_detail: str, wrong_reasoning_detail: str,
                 charge_matcher: ChargeMatcher,
                 law_str: str,
                 charges_to_mask: List[str], articles_to_mask: List[str],
                 true_charges: List[str], true_articles: List[str],
                 true_term: Dict, pred_term: Dict) -> Optional[Dict[str, Any]]:
    """
    Backtest: hide A/C labels, feed B/D reasoning, check if LLM self-corrects.
    Masks charge/article names in reasoning before feeding.
    """
    masked_correct = mask_labels_in_text(correct_reasoning_detail, charges_to_mask, articles_to_mask,
                                          true_term, pred_term)
    masked_wrong = mask_labels_in_text(wrong_reasoning_detail, charges_to_mask, articles_to_mask,
                                        true_term, pred_term)

    prompt = BACKTEST_WITH_REASONING_PROMPT.format(
        fact=fact,
        correct_reasoning=masked_correct,
        wrong_reasoning=masked_wrong,
        law=law_str,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        content = content.removeprefix("```json").removesuffix("```").strip()
        backtest_result = json.loads(content)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return None

    new_charges = set(charge_matcher.map_charges(backtest_result.get("charges", [])))
    new_articles = set(backtest_result.get("articles", []))
    new_term = backtest_result.get("term", {})

    true_charges_set = set(true_charges)
    true_articles_set = set(true_articles)

    # Determine controversy based on whether backtest recovers correct labels
    charges_match = new_charges == true_charges_set
    articles_match = new_articles == true_articles_set
    term_match = is_term_accurate(true_term, new_term)

    if charges_match and articles_match:
        controversy = 0.1  # Low controversy: fully self-corrected
        corrected = True
    elif charges_match or articles_match:
        controversy = 0.5  # Partial correction
        corrected = False
    else:
        controversy = 0.9  # High controversy: failed to correct
        corrected = False

    return {
        "charges": list(new_charges),
        "articles": list(new_articles),
        "term": new_term,
        "reasoning": backtest_result.get("reasoning", ""),
        "corrected": corrected,
        "controversy_score": controversy,
    }


def process_case(client: OpenAI, model: str, case: Dict[str, Any],
                 charge_matcher: ChargeMatcher, law_str: str) -> Optional[Dict[str, Any]]:
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

    # 2. L2 generation (cognitive layer: short + long reasoning + inspirations)
    analysis = run_l2_generation(client, model, fact, pred_reasoning,
                                 true_charges, true_articles,
                                 true_term, pred_term)
    if not analysis:
        return None

    correct_detail = analysis.get("correct_reasoning_detail", "")
    wrong_detail = analysis.get("wrong_reasoning_detail", "")

    # 3. Backtest verification: hide A/C, feed B/D
    charges_to_mask = list(set(true_charges + predicted_charges))
    articles_to_mask = list(set(a for a in (true_articles + predicted_articles) if a))

    backtest = run_backtest(
        client, model, fact,
        correct_detail, wrong_detail,
        charge_matcher, law_str,
        charges_to_mask, articles_to_mask,
        true_charges, true_articles, true_term, pred_term,
    )
    if not backtest:
        return None

    controversy = backtest["controversy_score"]

    # Build output with L0+L1+L2+L3
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
        "correct_reasoning_short": analysis.get("correct_reasoning_short", ""),
        "correct_reasoning_detail": correct_detail,
        "wrong_reasoning_short": analysis.get("wrong_reasoning_short", ""),
        "wrong_reasoning_detail": wrong_detail,
        "error_summary": analysis.get("error_summary", []),
        "inspirations": analysis.get("inspirations", []),
        "controversy_score": controversy,
        "backtest_result": {
            "charges": backtest["charges"],
            "articles": backtest["articles"],
            "term": backtest["term"],
            "reasoning": backtest["reasoning"],
            "corrected": backtest["corrected"],
        },
    }
    L3 = {}

    return {
        "L0": L0,
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "controversy_score": controversy,
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
    parser = argparse.ArgumentParser(description='Build multi-model collaborative KB with L1+L2+backtest')
    parser.add_argument('--config', type=str, default=str(ROOT_DIR / 'config/kb_building.yaml'))
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()

    # Load accusation/law lists
    accu_path = ROOT_DIR / "data/accu.txt"
    law_path = ROOT_DIR / "data/law.txt"
    if not accu_path.exists() or not law_path.exists():
        logger.error("accu.txt or law.txt not found in data/")
        sys.exit(1)
    charge_matcher = ChargeMatcher(str(accu_path))
    with open(law_path, 'r', encoding='utf-8') as f:
        law_str = f.read().strip()

    # Configs
    with open(args.config, 'r', encoding='utf-8') as f:
        kb_config = yaml.safe_load(f)
    with open(ROOT_DIR / 'config/config.yaml', 'r', encoding='utf-8') as f:
        global_config = yaml.safe_load(f)

    build_config = kb_config.get("hierarchical_build", {})
    input_path = args.input or build_config.get("input", str(ROOT_DIR / "data/negative_error_cases/collected_errors.json"))
    output_path = args.output or build_config.get("output_multi", str(ROOT_DIR / "data/negative_error_cases/collected_errors_hierarchical_multi.json"))
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
    low_controversy = 0
    high_controversy = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for orig_idx, case in indexed_cases:
            future = executor.submit(process_case, client, model_name, case, charge_matcher, law_str)
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

            if result["controversy_score"] <= 0.5:
                low_controversy += 1
            else:
                high_controversy += 1

            if pbar:
                pbar.update(1)

            if len(all_results) % 10 == 0:
                current_metadata = dict(metadata)
                current_metadata["total_cases"] = len(all_results)
                current_metadata["low_controversy"] = low_controversy
                current_metadata["high_controversy"] = high_controversy
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({"metadata": current_metadata, "cases": all_results}, f, ensure_ascii=False, indent=2)

    if pbar:
        pbar.close()

    final_metadata = dict(metadata)
    final_metadata["total_cases"] = len(all_results)
    final_metadata["low_controversy"] = low_controversy
    final_metadata["high_controversy"] = high_controversy
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"metadata": final_metadata, "cases": all_results}, f, ensure_ascii=False, indent=2)

    logger.info(f"Done! Saved {len(all_results)} cases to {output_path}")
    logger.info(f"Low controversy (<=0.5): {low_controversy}, High controversy (>0.5): {high_controversy}")


if __name__ == "__main__":
    main()
