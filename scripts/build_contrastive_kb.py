#!/usr/bin/env python3
"""
Contrastive Knowledge Base construction (pilot phase).
For charge errors (true=A, pred=B), retrieves a contrastive case (true=B) from
training set and generates a four-part L2 analysis (rule_a, rule_b, case_analysis, inspirations)
with a comparison_group field.

Based on: 对比库设计方案.md

Usage:
  python scripts/build_contrastive_kb.py --pilot [--max-samples N]
  python scripts/build_contrastive_kb.py --all [--max-samples N]

Output: data/negative_error_cases/contrastive_kb_pilot.json
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
from collections import defaultdict

import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

# Five pilot confusion pairs from the design doc (bidirectional)
# Each (true, pred) pair means: primary has true_charge=true, pred_charge=pred
# The reverse (pred, true) means: primary has true_charge=pred, pred_charge=true
PILOT_PAIRS = [
    ("寻衅滋事", "故意伤害"),
    ("故意伤害", "寻衅滋事"),
    ("诈骗", "合同诈骗"),
    ("合同诈骗", "诈骗"),
    ("抢劫", "盗窃"),
    ("盗窃", "抢劫"),
    ("交通肇事", "过失致人死亡"),
    ("过失致人死亡", "交通肇事"),
    ("贪污", "受贿"),
    ("受贿", "贪污"),
]

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

CONTRASTIVE_L2_PROMPT = """你是一位法律认知分析专家。你将看到一对容易混淆的罪名——"{true_charge}"和"{pred_charge}"，以及两个案件：

**Primary case**（真值为{true_charge}，但被错误预测为{pred_charge}）：
事实：{primary_fact}

**Contrastive case**（真值为{pred_charge}的正确案例，用于对比）：
事实摘要：{contrastive_summary}
法律要素：{contrastive_elements_json}

请完成以下四项分析，输出严格JSON格式：

**1. rule_a**：在什么情况下应该判"{true_charge}"
   - 抽象的构成要件 + 典型场景描述
   - 不要绑定本案或对比案的具体细节，写通用规则

**2. rule_b**：在什么情况下应该判"{pred_charge}"
   - 抽象的构成要件 + 典型场景描述
   - 不要绑定本案或对比案的具体细节，写通用规则

**3. case_analysis**：本案（primary case）为什么符合rule_a的条件，而不符合rule_b的条件
   - 将具体事实映射到rule_a和rule_b，完成法律三段论
   - 明确指出本案事实中哪些特征指向{true_charge}，哪些特征不符合{pred_charge}

**4. inspirations**：1-2条认知启发
   - 模糊化、不确定性语言（"或许可以留意..."、"建议审视..."）
   - 脱离具体案件细节，只保留抽象思维方向
   - 不暗示任何判决结论

输出格式：
{{
  "rule_a": "...",
  "rule_b": "...",
  "case_analysis": "...",
  "inspirations": ["启发1", "启发2"]
}}

仅输出JSON，不要任何其他内容。"""


def load_api_config(global_config: dict):
    api_config = global_config.get("api", {})
    base_url = api_config.get("base_url")
    api_key_env_var = api_config.get("api_key", "DEEPSEEK_API_KEY")
    model_name = api_config.get("model_name")
    api_key = os.getenv(api_key_env_var)
    if not all([base_url, api_key, model_name]):
        raise ValueError(f"Missing API config: base_url, {api_key_env_var}, model_name.")
    return base_url, api_key, model_name


def load_train_index(train_path: str, max_per_charge: int = 500) -> Dict[str, List[Dict]]:
    """Load train.jsonl and index by accusation (charge name), sampling max_per_charge per charge."""
    import random
    raw_index = defaultdict(list)
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fact = obj.get("fact", "")
            meta = obj.get("meta", {})
            charges = meta.get("accusation", [])
            for c in charges:
                raw_index[c].append({
                    "fact": fact,
                    "charges": charges,
                    "articles": meta.get("relevant_articles", []),
                    "term": meta.get("term_of_imprisonment", {}),
                    "fine": meta.get("punish_of_money", 0),
                })

    # Sample max_per_charge per charge for memory/speed
    index = {}
    for c, cases in raw_index.items():
        if len(cases) > max_per_charge:
            random.shuffle(cases)
            cases = cases[:max_per_charge]
        index[c] = cases

    logger.info(f"Loaded train index: {len(index)} charges, {sum(len(v) for v in index.values())} cases (max {max_per_charge}/charge)")
    return index


def find_best_contrastive(train_index: Dict[str, List[Dict]],
                          model: SentenceTransformer,
                          pred_charge: str,
                          primary_fact: str) -> Optional[Dict]:
    """Find the most fact-similar contrastive case for pred_charge using Sentence-BERT."""
    candidates = train_index.get(pred_charge, [])
    if not candidates:
        logger.warning(f"No contrastive candidates found for charge: {pred_charge}")
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Sample max 200 candidates for speed
    if len(candidates) > 200:
        import random
        candidates = random.sample(candidates, 200)

    # Encode primary fact
    primary_emb = model.encode([primary_fact], normalize_embeddings=True)[0]

    # Encode candidate facts in batches
    candidate_facts = [c["fact"] for c in candidates]
    candidate_embs = model.encode(candidate_facts, normalize_embeddings=True)

    # Compute similarities
    similarities = np.dot(candidate_embs, primary_emb)
    best_idx = int(np.argmax(similarities))
    best = candidates[best_idx]
    best["contrastive_similarity"] = float(similarities[best_idx])
    logger.info(f"Best contrastive for {pred_charge}: similarity={best['contrastive_similarity']:.3f}")
    return best


def extract_elements(client: OpenAI, model_name: str, fact: str) -> Optional[Dict[str, str]]:
    """Extract 7 legal elements from case fact."""
    prompt = ELEMENT_EXTRACTION_PROMPT.format(fact=fact)
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        content = content.removeprefix("```json").removesuffix("```").strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Element extraction failed: {e}")
        return None


def generate_contrastive_l2(client: OpenAI, model_name: str,
                            true_charge: str, pred_charge: str,
                            primary_fact: str,
                            contrastive_summary: str,
                            contrastive_elements: Dict) -> Optional[Dict]:
    """Generate L2 with contrastive analysis (rule_a, rule_b, case_analysis, inspirations)."""
    elements_json = json.dumps(contrastive_elements, ensure_ascii=False, indent=2)
    prompt = CONTRASTIVE_L2_PROMPT.format(
        true_charge=true_charge,
        pred_charge=pred_charge,
        primary_fact=primary_fact,
        contrastive_summary=contrastive_summary[:300],
        contrastive_elements_json=elements_json,
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        content = content.removeprefix("```json").removesuffix("```").strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Contrastive L2 generation failed: {e}")
        return None


def get_core_confusion(true_charges: List[str], pred_charges: List[str],
                       pilot_pairs_set: set) -> Optional[Tuple[str, str]]:
    """Extract the core confusion pair from multi-charge cases.
    Priority: match pilot pairs first, then find any true≠pred pair."""
    for t in true_charges:
        for p in pred_charges:
            if (t, p) in pilot_pairs_set:
                return (t, p)
    # Fallback: first true≠pred pair
    for t in true_charges:
        if t not in pred_charges:
            for p in pred_charges:
                if p != t:
                    return (t, p)
    return None


def _format_term(term: dict) -> str:
    if term.get("death_penalty"):
        return "死刑"
    if term.get("life_imprisonment"):
        return "无期徒刑"
    imp = term.get("imprisonment", 0)
    if imp > 0:
        return f"有期徒刑{imp}个月"
    return "无刑期"


def process_charge_error(client: OpenAI, model_name: str,
                         case: Dict, train_index: Dict,
                         emb_model: SentenceTransformer,
                         pilot_pairs_set: set,
                         pilot_mode: bool) -> Optional[Dict]:
    """Process a single charge error with contrastive retrieval and L2 generation."""
    true_charges = case.get("true_charges", [])
    pred_charges = case.get("predicted_charges", [])
    fact = case.get("fact", "")

    if not fact or not true_charges or not pred_charges:
        return None

    # Get core confusion pair
    confusion = get_core_confusion(true_charges, pred_charges, pilot_pairs_set)
    if not confusion:
        return None

    true_charge, pred_charge = confusion

    if pilot_mode and confusion not in pilot_pairs_set:
        return None

    # Find contrastive case from training set
    # Build a short summary (first ~200 chars) of primary fact for similarity search
    contrastive = find_best_contrastive(train_index, emb_model, pred_charge, fact)
    if not contrastive:
        logger.warning(f"No contrastive case found for {pred_charge}, skipping")
        return None

    contrastive_fact = contrastive["fact"]

    # Extract L1 elements for contrastive case
    logger.info(f"Extracting L1 elements for contrastive case ({pred_charge})...")
    contrastive_elements = extract_elements(client, model_name, contrastive_fact)
    if not contrastive_elements:
        logger.error(f"Failed to extract elements for contrastive case, skipping")
        return None

    # Generate L2 with contrastive prompt
    logger.info(f"Generating contrastive L2 for {true_charge} vs {pred_charge}...")
    contrastive_summary = contrastive_fact[:200]
    l2_result = generate_contrastive_l2(
        client, model_name,
        true_charge, pred_charge,
        fact, contrastive_summary, contrastive_elements,
    )
    if not l2_result:
        return None

    # Build output entry
    L0 = {
        "fact": fact,
        "true_charges": true_charges,
        "predicted_charges": pred_charges,
        "true_articles": case.get("true_articles", []),
        "predicted_articles": case.get("predicted_articles", []),
        "true_term": case.get("true_term", {}),
        "predicted_term": case.get("predicted_term", {}),
        "true_fine": case.get("true_fine", 0),
        "predicted_fine": case.get("predicted_fine", 0),
        "pred_reasoning": case.get("pred_reasoning", ""),
    }
    L1 = {"legal_elements": {}}  # Primary's L1 not extracted here (use existing if available)

    # Build comparison group
    comparison_group = {
        "confusion_pair": [true_charge, pred_charge],
        "contrastive_entries": [
            {
                "charge": pred_charge,
                "case_summary": contrastive_fact[:150],
                "legal_elements": contrastive_elements,
                "source": "train.json",
            }
        ],
    }

    entry = {
        "entry_type": "error",
        "confusion_pair": [true_charge, pred_charge],
        "L0": L0,
        "L1": L1,
        "L2": {
            "rule_a": l2_result.get("rule_a", ""),
            "rule_b": l2_result.get("rule_b", ""),
            "case_analysis": l2_result.get("case_analysis", ""),
            "inspirations": l2_result.get("inspirations", []),
            "case_summary": "",
            "correct_reasoning_short": "",
            "correct_reasoning_detail": "",
            "wrong_reasoning_short": "",
            "wrong_reasoning_detail": "",
            "error_summary": [f"Charge error: true={true_charge}, pred={pred_charge}"],
        },
        "comparison_group": comparison_group,
    }
    return entry


def main():
    parser = argparse.ArgumentParser(description="Build contrastive KB (pilot)")
    parser.add_argument("--config", type=str, default=str(ROOT_DIR / "config/kb_building.yaml"))
    parser.add_argument("--pilot", action="store_true", help="Only process 5 pilot confusion pairs")
    parser.add_argument("--all", action="store_true", help="Process all charge errors")
    parser.add_argument("--max-samples", type=int, default=None, help="Max charge errors to process")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--output", type=str,
                        default=str(ROOT_DIR / "data/negative_error_cases/contrastive_kb_pilot.json"))
    args = parser.parse_args()

    if not args.pilot and not args.all:
        parser.print_help()
        print("\nError: specify --pilot or --all")
        sys.exit(1)

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        kb_config = yaml.safe_load(f)
    with open(ROOT_DIR / 'config/config.yaml', 'r', encoding='utf-8') as f:
        global_config = yaml.safe_load(f)

    # API setup
    base_url, api_key, model_name = load_api_config(global_config)
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Build pilot pairs set (must be defined before filtering)
    pilot_pairs_set = set(PILOT_PAIRS)

    # Load collected errors
    input_path = kb_config.get("collection", {}).get("output",
                     str(ROOT_DIR / "data/negative_error_cases/collected_errors.json"))
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    all_cases = input_data.get("error_cases", [])
    logger.info(f"Loaded {len(all_cases)} total error cases")

    # Filter charge errors
    charge_errors = []
    for c in all_cases:
        tc = sorted(c.get("true_charges", []))
        pc = sorted(c.get("predicted_charges", []))
        if tc != pc:
            charge_errors.append(c)
    logger.info(f"Found {len(charge_errors)} charge errors")

    # Filter to pilot pairs first (if requested), THEN apply max-samples
    if args.pilot:
        initial_count = len(charge_errors)
        charge_errors = [c for c in charge_errors if get_core_confusion(
            c.get("true_charges", []), c.get("predicted_charges", []), pilot_pairs_set) in pilot_pairs_set]
        logger.info(f"Filtered to {len(charge_errors)} pilot cases (from {initial_count} charge errors)")

    if args.max_samples:
        charge_errors = charge_errors[:args.max_samples]
        logger.info(f"Limited to {len(charge_errors)} charge errors")

    # Load train index
    train_path = kb_config.get("collection", {}).get("train_file",
                     str(ROOT_DIR / "data/final_all_data/first_stage/train.json"))
    train_index = load_train_index(train_path)

    # Load embedding model for contrastive retrieval
    logger.info("Loading Sentence-BERT model for contrastive retrieval...")
    emb_model = SentenceTransformer("uer/sbert-base-chinese-nli", device="cpu")


    if not charge_errors:
        logger.warning("No charge errors to process after filtering")
        return

    # Process cases
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = []
    errors = 0

    for i, case in enumerate(charge_errors):
        logger.info(f"[{i+1}/{len(charge_errors)}] Processing charge error...")
        result = process_charge_error(
            client, model_name, case, train_index, emb_model,
            pilot_pairs_set, pilot_mode=args.pilot,
        )
        if result is None:
            errors += 1
            continue
        results.append(result)

        # Save intermediate results every 1 case (pilot is small)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "mode": "contrastive_kb_pilot" if args.pilot else "contrastive_kb_full",
                    "total_entries": len(results),
                    "errors": errors,
                    "pilot_pairs": PILOT_PAIRS if args.pilot else [],
                },
                "entries": results,
            }, f, ensure_ascii=False, indent=2)

    logger.info(f"Done! Saved {len(results)} entries to {args.output}")
    if errors:
        logger.warning(f"{errors} errors during processing")


if __name__ == "__main__":
    main()
