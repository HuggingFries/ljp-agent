#!/usr/bin/env python3
"""
Build positive hierarchical knowledge base from collected positive cases.
Reads API config from config/config.yaml.

Usage:
  python scripts/build_hierarchical_positive_kb.py [--input INPUT] [--output OUTPUT] [--max-workers N] [--resume-from OUTPUT]
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
from pathlib import Path

import yaml
from openai import OpenAI

try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_api_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load API configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    api = config.get('api', {})
    # Support reading api_key from environment variable if placeholder present
    api_key = api.get('api_key', '')
    if api_key == 'DEEPSEEK_API_KEY' or api_key.startswith('${'):
        api_key = os.getenv('DEEPSEEK_API_KEY', '')
    api['api_key'] = api_key
    return api


def build_extraction_prompt(item: Dict[str, Any]) -> str:
    """使用中文 prompt 提取正例的 L1 和 L2 信息"""
    fact = item.get('fact', '').strip()
    true_charges = item.get('true_charges', [])
    true_articles = item.get('true_articles', [])
    charge_single = item.get('charge', true_charges[0] if true_charges else '')

    prompt = f"""你是一位法律AI专家。请分析以下正确判决的案例，并提取结构化信息。

## 案件事实
{fact}

## 真实罪名（正确）
{', '.join(true_charges) if true_charges else '无'}

## 真实法条
{', '.join(true_articles) if true_articles else '无'}

## 提取要求
请严格按照下面的JSON格式输出，不要添加任何额外内容。
{{
"L1": {{
"犯罪主体": "抽象描述：例如 '单人/多人/单位；身份特点（如国家工作人员、普通公民）；是否有前科等（只写性质，不写姓名）'",
"犯罪行为": "行为的性质类型：例如 '秘密窃取/暴力胁迫/欺诈骗取/非法持有'",
"犯罪手段": "实施行为的手段特点：例如 '持刀/投放危险物质/利用信息网络'",
"犯罪客体": "行为侵犯的客体类别：例如 '公共安全/公民人身权利/财产权利'",
"犯罪动机": "主观罪过形式（故意/过失）及动机：例如 '故意，图财' 或 '过失'",
"危害程度": "危害结果的严重程度：例如 '造成死亡/轻伤/数额较大/数额巨大'",
"法益类型": "具体侵犯的法益类型：例如 '盗窃罪侵犯财产所有权；故意伤害侵犯身体权'"
}},
"L2": {{
"key_facts_summary": "一句话概括对判决最关键的法律事实（最多50字）。去除姓名、日期、程序性信息，只保留直接影响罪名和法条选择的事实。",
"judgment_reasoning": "完整的三段论推理：(1) 大前提：适用的法条及其构成要件（例如《刑法》第X条规定……）；(2) 小前提：本案事实如何满足每一个构成要件；(3) 结论：因此，被告人犯[罪名]，应依照[法条]处罚。"
}}
}}

text

## 重要说明

1. **L1 各字段**：所有描述必须**抽象**，不出现具体姓名、地名、金额、时间等。使用法律要素的分类描述。
2. **L2.key_facts_summary**：极简，只保留法律上关键的事实，忽略庭审程序、证人、日期等。
3. **L2.judgment_reasoning**：必须遵循三段论格式。大前提要引用正确的法条并简述构成要件；小前提将事实与要件一一对应；结论给出最终罪名和法条。
4. 输出**只包含合法的 JSON**，不要有其他解释或 Markdown 标记。

现在请生成 JSON。
"""
    return prompt


def process_positive_case(
    client: OpenAI,
    model_name: str,
    item: Dict[str, Any],
    idx: int
) -> Optional[Dict[str, Any]]:
    """Process one positive case: call LLM and return hierarchical structure."""
    prompt = build_extraction_prompt(item)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        logger.error(f"API call failed for index {idx}: {e}")
        return None

    content = response.choices[0].message.content
    usage = response.usage

    # Parse JSON
    try:
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)
        parsed = json.loads(content)
    except Exception as e:
        logger.warning(f"JSON parsing failed for index {idx}: {e}\nContent: {content[:200]}...")
        return None

    result = {
        "L0": {
            "fact": item.get("fact", ""),
            "true_charges": item.get("true_charges", []),
            "true_articles": item.get("true_articles", []),
            "charge": item.get("charge", ""),
        },
        "L1": parsed.get("L1", {}),
        "L2": parsed.get("L2", {}),
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0
        }
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Build positive hierarchical knowledge base")
    parser.add_argument("--input", default="data/positive_cases/collected_positive_cases.json",
                        help="Input positive cases JSON")
    parser.add_argument("--output", default="data/positive_cases/collected_positive_hierarchical.json",
                        help="Output hierarchical JSON")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Max parallel API workers")
    parser.add_argument("--resume-from", help="Resume from existing output file")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Config YAML file (contains API section)")
    args = parser.parse_args()

    # Load API config
    try:
        api_config = load_api_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    base_url = api_config.get("base_url", "https://api.deepseek.com/v1")
    api_key = api_config.get("api_key")
    model_name = api_config.get("model_name", "deepseek-chat")

    if not api_key:
        logger.error("API key not found. Set DEEPSEEK_API_KEY environment variable or provide in config.yaml")
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Load positive cases
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    logger.info(f"Loaded {len(cases)} positive cases from {input_path}")

    # Handle resume
    processed_indices = set()
    results = []
    if args.resume_from and Path(args.resume_from).exists():
        logger.info(f"Resuming from {args.resume_from}")
        with open(args.resume_from, 'r', encoding='utf-8') as f:
            resume_data = json.load(f)
        if isinstance(resume_data, list):
            for processed in resume_data:
                orig_idx = processed.get("L0", {}).get("original_index")
                if orig_idx is not None:
                    processed_indices.add(orig_idx)
                    results.append(processed)
            logger.info(f"Resumed {len(results)} already processed cases")
        elif isinstance(resume_data, dict) and "cases" in resume_data:
            for processed in resume_data["cases"]:
                orig_idx = processed.get("L0", {}).get("original_index")
                if orig_idx is not None:
                    processed_indices.add(orig_idx)
                    results.append(processed)
            logger.info(f"Resumed {len(results)} already processed cases")

    # Prepare unprocessed
    unprocessed = []
    for idx, case in enumerate(cases):
        if idx not in processed_indices:
            case["original_index"] = idx
            unprocessed.append(case)

    logger.info(f"{len(unprocessed)} cases remaining to process")
    if not unprocessed:
        logger.info("All cases already processed, exiting")
        return

    # Process in parallel
    total = len(unprocessed)
    completed = 0
    pbar = tqdm(total=total) if has_tqdm else None

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_positive_case, client, model_name, item, item["original_index"])
                   for item in unprocessed]

        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res:
                results.append(res)
            else:
                logger.warning(f"Failed to process some case (completed {completed}/{total})")

            if pbar:
                pbar.update(1)

            # Periodic save
            if completed % 10 == 0:
                logger.info(f"Processed {completed}/{total}, saving intermediate...")
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    if pbar:
        pbar.close()

    # Final save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Done! Saved {len(results)} processed positive cases to {args.output}")
    failed = len(unprocessed) - (len(results) - len(processed_indices))
    if failed > 0:
        logger.warning(f"{failed} cases failed to process. Re-run with --resume-from to retry.")


if __name__ == "__main__":
    main()