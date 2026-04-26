#!/usr/bin/env python3
"""
Refined cognitive distillation for legal error cases.
Outputs L0/L1/L2/L3 hierarchical structure with mild "inspirations"
instead of absolute cognitive capsules.

Stage 1: Extract L1 legal elements (seven elements) using LLM.
Stage 2: Generate cognitive inspirations (mild, non-directive) from error case.
Stage 3: Backtest with inspirations, compute controversy score.
L3 is reserved (empty).

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
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
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

INSPIRATION_GENERATION_PROMPT = """你是一位法律认知启发提炼专家。你将看到一个案件的错误预测信息。你的任务是从中提炼出1-3条**认知启发**，帮助其他法律AI在遇到类似推理情境时，能更谨慎、更全面地思考。

**背景信息**：
案件事实：
{fact}

模型的错误预测及推理过程：
{pred_reasoning}

正确的判决结果：{true_charges}（法条：{true_articles}）

**要求**：
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

4. 启发不能暗示任何判决结论，也不能暗示哪种思考方向是“正确”的。它只负责温和地提醒模型去审视某个容易被忽略的维度，但不告诉模型审视的结果应该是什么。

5. 即使模型正在审理的案件与当前错误案例完全不同，这些启发也应该是**无害的**——它们只是温和的提醒，不会误导或强迫模型。

**示例**（好的启发）：
- "当案件涉及数个相互关联的行为时，或许可以审视这些行为是否被同一概括故意所覆盖。"
- "对于涉及数额门槛的罪名，是否需要确认具体数额是否达到了相应司法解释的标准？"
- "在共同犯罪中，建议审视各参与人的行为是否可能构成不同的罪名。"

**示例**（坏的启发，不要这样写）：
- "必须审查构成要件，不能直接套用熟悉罪名。"
- "禁止忽略对数额门槛的审查。"
- "正确做法是先检查竞合关系。"

请输出一个JSON数组，每个元素是一个字符串，表示一条启发。

输出格式（严格JSON）：
{{
  "inspirations": [
    "启发1",
    "启发2"
  ]
}}

仅输出JSON，不要添加任何其他内容。"""

STAGE3_BACKTEST_PROMPT = """你是一位法律AI。请根据以下案件事实，结合提供的**认知启发**，在给定的罪名列表和法条列表中选择正确的罪名和法条进行判决预测。

案件事实：
{fact}

待选罪名列表：{accu}
待选法条列表：{law}

认知启发（温和的思维提醒，可能并不都与本案直接相关，仅供参考）：
{inspirations_text}

请基于以上指导，输出：
1. 新的推理过程
2. 新的预测结果（罪名+法条）

以JSON格式输出：
{{
  "new_reasoning": "...",
  "new_charges": ["罪名1"],
  "new_articles": ["法条1"]
}}

仅输出JSON。"""


# --------------------------- Helper Functions ---------------------------

def run_l1_extraction(client: OpenAI, model: str, fact: str) -> Optional[Dict[str, str]]:
    """提取七要素（L1）"""
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


def run_inspiration_generation(client: OpenAI, model: str, fact: str, pred_reasoning: str,
                               true_charges: List[str], true_articles: List[str]) -> Optional[List[str]]:
    """Generate mild cognitive inspirations from an error case."""
    prompt = INSPIRATION_GENERATION_PROMPT.format(
        fact=fact,
        pred_reasoning=pred_reasoning,
        true_charges="、".join(true_charges) if true_charges else "无",
        true_articles="、".join(true_articles) if true_articles else "无"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        content = content.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(content)
        return data.get("inspirations", [])
    except Exception as e:
        logger.error(f"Inspiration generation failed: {e}")
        return None


def format_inspirations(inspirations: List[str]) -> str:
    """Format inspirations for prompt injection."""
    if not inspirations:
        return ""
    lines = []
    for i, insp in enumerate(inspirations, 1):
        lines.append(f"{i}. {insp}")
    return "\n".join(lines)


def run_stage3(client: OpenAI, model: str, fact: str,
               inspirations_text: str,
               original_prediction: Dict[str, list],
               true_charges: List[str], true_articles: List[str],
               accu: str, law: str) -> Optional[Dict[str, Any]]:
    """Backtest with inspirations and compute controversy score."""
    prompt = STAGE3_BACKTEST_PROMPT.format(
        fact=fact,
        accu=accu,
        law=law,
        inspirations_text=inspirations_text
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
        logger.error(f"Stage3 backtest failed: {e}")
        return None

    new_charges = backtest_result.get("new_charges", [])
    new_articles = backtest_result.get("new_articles", [])
    old_charges = original_prediction.get("predicted_charges", [])
    old_articles = original_prediction.get("predicted_articles", [])

    if set(new_charges) == set(true_charges) and set(new_articles) == set(true_articles):
        controversy = 0.1  # Fully corrected
    elif set(new_charges) != set(old_charges) or set(new_articles) != set(old_articles):
        controversy = 0.5  # Changed but not fully correct
    else:
        controversy = 0.9  # No change

    return {
        "new_charges": new_charges,
        "new_articles": new_articles,
        "new_reasoning": backtest_result.get("new_reasoning", ""),
        "controversy_score": controversy,
    }


def process_case(client: OpenAI, model: str, case: Dict[str, Any],
                 accu_str: str, law_str: str) -> Optional[Dict[str, Any]]:
    fact = case.get("fact", "")
    pred_reasoning = case.get("pred_reasoning", "")
    predicted_charges = case.get("predicted_charges", [])
    true_charges = case.get("true_charges", [])
    true_articles = case.get("true_articles", [])
    predicted_articles = case.get("predicted_articles", [])

    if not fact or not pred_reasoning:
        logger.warning("Missing fact or reasoning, skipping")
        return None

    # 1. L1 extraction
    l1_elements = run_l1_extraction(client, model, fact)
    if not l1_elements:
        return None

    # 2. Generate inspirations (without confusion pairs)
    inspirations = run_inspiration_generation(client, model, fact, pred_reasoning,
                                               true_charges, true_articles)
    if not inspirations:
        return None
    inspirations_text = format_inspirations(inspirations)

    # 3. Backtest with inspirations only
    original_pred = {
        "predicted_charges": predicted_charges,
        "predicted_articles": predicted_articles,
    }
    backtest = run_stage3(client, model, fact, inspirations_text,
                          original_pred, true_charges, true_articles,
                          accu_str, law_str)
    if not backtest:
        return None

    controversy = backtest["controversy_score"]

    # Build output with modified L2
    L0 = {
        "fact": fact,
        "true_charges": true_charges,
        "predicted_charges": predicted_charges,
        "true_articles": true_articles,
        "predicted_articles": predicted_articles,
        "pred_reasoning": pred_reasoning,
        "predict_prompt_tokens": case.get("predict_prompt_tokens", 0),
        "predict_completion_tokens": case.get("predict_completion_tokens", 0),
    }
    L1 = {"legal_elements": l1_elements}
    L2 = {
        "inspirations": inspirations,                      # mild, non-directive
        "controversy_score": controversy,
        "backtest_new_charges": backtest["new_charges"],
        "backtest_new_articles": backtest.get("new_articles", []),
        "backtest_new_reasoning": backtest.get("new_reasoning", ""),
        "original_predicted_charges": original_pred["predicted_charges"],
        "original_predicted_articles": original_pred["predicted_articles"],
        "true_charges": true_charges,
        "true_articles": true_articles,
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
    parser = argparse.ArgumentParser(description='Build refined negative KB with cognitive inspirations')
    parser.add_argument('--config', type=str, default=str(ROOT_DIR / 'config/kb_building.yaml'))
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()

    # Load accusation/law lists for backtest
    accu_path = ROOT_DIR / "data/accu.txt"
    law_path = ROOT_DIR / "data/law.txt"
    if not accu_path.exists() or not law_path.exists():
        logger.error("accu.txt or law.txt not found in data/")
        sys.exit(1)
    with open(accu_path, 'r', encoding='utf-8') as f:
        accu_str = f.read().strip()
    with open(law_path, 'r', encoding='utf-8') as f:
        law_str = f.read().strip()

    # Configs
    with open(args.config, 'r', encoding='utf-8') as f:
        kb_config = yaml.safe_load(f)
    with open(ROOT_DIR / 'config/config.yaml', 'r', encoding='utf-8') as f:
        global_config = yaml.safe_load(f)

    build_config = kb_config.get("hierarchical_build", {})
    input_path = args.input or build_config.get("input", str(ROOT_DIR / "data/negative_error_cases/collected_errors.json"))
    output_path = args.output or build_config.get("output_multi", str(ROOT_DIR / "data/negative_error_cases/collected_errors_hierarchical_multi_refined.json"))
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
            future = executor.submit(process_case, client, model_name, case, accu_str, law_str)
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
    logger.info(f"Low controversy (≤0.5): {low_controversy}, High controversy (>0.5): {high_controversy}")


if __name__ == "__main__":
    main()