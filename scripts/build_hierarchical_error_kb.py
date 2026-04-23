#!/usr/bin/env python3
"""
Build hierarchical structured negative knowledge base from collected error cases.

Process each raw error case (L0) with LLM to generate hierarchical structure:
- L0: Original raw data (passthrough)
- L1: Legal elements layer (for retrieval, extracted with same method as element_extractor.py)
- L2: Case-specific error analysis layer (for prompt injection)
- L3: Generalized experience/lessons layer (for prompt injection)

Usage:
  python build_hierarchical_error_kb.py [options]

Options:
  --config CONFIG       Config file path (default: config/kb_building.yaml)
  --input PATH          Input collected raw errors (default: data/negative_error_cases/collected_errors.json)
  --output PATH         Output hierarchical json (default: data/negative_error_cases/collected_errors_hierarchical.json)
  --max-workers N       Parallel API workers (default: 10)
  --resume-from PATH    Resume from existing output
"""

import argparse
import json
import logging
import random
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
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

# Project paths
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

# Import element extraction prompt from the shared module
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

仅输出JSON，不要输出任何其他内容。
"""


def build_analysis_prompt(item: Dict[str, Any]) -> str:
    """Build the LLM prompt for L2 + L3 hierarchical analysis"""
    fact = item.get('fact', '').strip()
    pred_reasoning = item.get('pred_reasoning', '').strip()
    true_charges = item.get('true_charges', [])
    predicted_charges = item.get('predicted_charges', [])
    true_articles = item.get('true_articles', [])
    predicted_articles = item.get('predicted_articles', [])
    
    prompt = f"""你是一位资深法律AI专家，请帮我对这个模型预测错误的刑事案件，进行结构化错误分析，生成指定格式的L2和L3两层内容。

## 原始案件信息
### 案件事实
{fact}

### 模型原始预测推理过程（模型为什么这么判）
{pred_reasoning}

### 真实罪名（正确判决）
{', '.join(true_charges)}

### 模型预测错误罪名
{', '.join(predicted_charges)}

### 真实法条编号（正确）
{', '.join(true_articles)}

### 模型预测错误法条编号
{', '.join(predicted_articles)}

## 分析要求
请严格按照下面的JSON格式输出，不要添加任何额外内容：

```json
{{
  "L2": {{
    "case_summary": "案件关键事实的概括，简洁明了",
    "correct_reasoning": "应该判处true_charges和引用true_articles的原因，用法律三段论格式：大前提是法条内容，小前提是本案事实，结论是本案构成该罪。",
    "wrong_reasoning": "不该判处predicted_charges和引用predicted_articles的原因，同样用三段论格式。",
    "error_summary": ["本案涉及的错误汇总，如果涉及多个错误，分条写出"],
    "controversy_score": 0.0
  }},
  "L3": {{
    "experience": "从本案提炼的**抽象普适性**经验，模型应当遵守，不提及本案，通用即可。如果没有特别经验，留空字符串",
    "lesson": "从本案提炼的**抽象普适性**教训，模型应该避免，不提及本案，通用即可。如果没有特别教训，留空字符串",
    "hint": "更宽泛的提示，告诉模型遇到类似情况要注意什么，比如\"盗窃罪和诈骗罪容易混淆，需要根据被害人是否基于错误认识处分财物来区分\"，不要写成命令形式。不提及本案，通用即可。如果没有特别提示，留空字符串"
  }}
}}
```

## 重要提示
1. **分层分工**：L2是针对本案的具体分析，L3必须是**抽象普适性**的经验教训，绝对不能提及本案，方便复用给其他类似案件。
2. 你必须参考模型给出的`pred_reasoning`（模型为什么这么判）来分析错误，不要自己瞎猜错误原因，模型已经告诉你它是怎么推理的了。
3. 三段论格式必须严格遵守：大前提（法条内容） → 小前提（本案事实） → 结论（定性）。
4. **controversy_score争议性分数定义**：给本案的争议程度打分，范围0~1：
   - 0.0 ~ 0.3：事实清楚，法律适用明确，结论没有争议
   - 0.4 ~ 0.6：存在一定争议，但主流观点倾向于结论
   - 0.7 ~ 1.0：事实模糊，法律适用存在较大分歧，或者本身属于司法解释漏洞，数据标注可能本身就有问题，结论本身可商榷
   分数越高，代表本案结论越有争议，后续RAG会根据分数加权，低争议案例权重更高。
5. **关于案件事实中的罪名提示**：CAIL2018数据中，案件事实里有时候会出现罪名（如"被告人涉嫌盗窃罪"），这是公诉方指控的罪名，不是法院最终判决结论，法院不一定采纳，最终结论以我们给出的`true_charges`为准。分析时请注意区分，不要被事实中出现的公诉罪名误导。
6. L3必须通用：绝对不能出现"本案中""本案""该案"等个案相关描述，只写通用规则知识，避免给不相关案件带来噪声。
7. L3只填写错误涉及的部分，不涉及的部分留空字符串，不要硬凑内容。
8. 必须合法合规输出严格JSON格式，不要有语法错误。
"""
    return prompt


def extract_hierarchical_structure(
    client: OpenAI,
    model_name: str,
    item: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Extract hierarchical structure from one raw error case using LLM"""
    # Step 1: Extract L1 legal elements (using same prompt as element_extractor.py)
    fact = item.get('fact', '')
    
    try:
        # Extract L1
        l1_prompt = ELEMENT_EXTRACTION_PROMPT.format(fact=fact)
        l1_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": l1_prompt}],
            temperature=0.0,
        )
        l1_content = l1_response.choices[0].message.content.strip()
        l1_content = l1_content.removeprefix("```json").removesuffix("```").strip()
        l1_elements = json.loads(l1_content)
        l1_usage = l1_response.usage
        
        # Step 2: Extract L2 + L3 analysis
        l2_prompt = build_analysis_prompt(item)
        l2_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": l2_prompt}],
            temperature=0.0,
        )
        l2_content = l2_response.choices[0].message.content.strip()
        l2_content = l2_content.removeprefix("```json").removesuffix("```").strip()
        l2l3_parsed = json.loads(l2_content)
        l2_usage = l2_response.usage
        
        total_prompt = (l1_usage.prompt_tokens if l1_usage else 0) + (l2_usage.prompt_tokens if l2_usage else 0)
        total_completion = (l1_usage.completion_tokens if l1_usage else 0) + (l2_usage.completion_tokens if l2_usage else 0)
        
    except Exception as e:
        logger.error(f"Failed to extract structure: {e}")
        return None
    
    # Build final result with all layers
    result = {
        "L0": {
            "fact": item.get("fact", ""),
            "true_charges": item.get("true_charges", []),
            "predicted_charges": item.get("predicted_charges", []),
            "true_articles": item.get("true_articles", []),
            "predicted_articles": item.get("predicted_articles", []),
            "pred_reasoning": item.get("pred_reasoning", ""),
            "predict_prompt_tokens": item.get("predict_prompt_tokens", 0),
            "predict_completion_tokens": item.get("predict_completion_tokens", 0),
        },
        "L1": {
            "legal_elements": l1_elements
        },
        "L2": l2l3_parsed.get("L2", {}),
        "L3": l2l3_parsed.get("L3", {}),
        "usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion
        }
    }
    
    return result


def load_api_config(global_config: dict):
    """Load API configuration from global config"""
    api_config = global_config.get("api", {})
    base_url = api_config.get("base_url")
    api_key_env_var = api_config.get("api_key", "DEEPSEEK_API_KEY")
    model_name = api_config.get("model_name")
    
    api_key = os.getenv(api_key_env_var)
    
    if not all([base_url, api_key, model_name]):
        raise ValueError(
            f"Missing API configuration:\n"
            f"  - base_url: {base_url}\n"
            f"  - api_key from env '{api_key_env_var}': {api_key is not None}\n"
            f"  - model_name: {model_name}\n"
            f"Please check config/config.yaml and environment variable."
        )
    
    return base_url, api_key, model_name


def main():
    parser = argparse.ArgumentParser(description='Build hierarchical negative KB from raw collected errors')
    parser.add_argument('--config', type=str, default=str(ROOT_DIR / 'config/kb_building.yaml'), help='Config file path')
    parser.add_argument('--input', type=str, default=None, help='Input raw collected errors json path')
    parser.add_argument('--output', type=str, default=None, help='Output hierarchical json path')
    parser.add_argument('--max-workers', type=int, default=None, help='Parallel API workers')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from existing output')
    parser.add_argument('--max-samples', type=int, default=None, help='Only process first N samples for test')
    args = parser.parse_args()
    
    # Read kb building config
    with open(args.config, 'r', encoding='utf-8') as f:
        kb_config = yaml.safe_load(f)
    
    # Read global config for API
    global_config_path = ROOT_DIR / 'config/config.yaml'
    with open(global_config_path, 'r', encoding='utf-8') as f:
        global_config = yaml.safe_load(f)
    
    # Get paths from config, command line overrides
    build_config = kb_config.get("hierarchical_build", {})
    input_path = args.input or build_config.get("input", str(ROOT_DIR / "data/negative_error_cases/collected_errors.json"))
    output_path = args.output or build_config.get("output", str(ROOT_DIR / "data/negative_error_cases/collected_errors_hierarchical.json"))
    max_workers = args.max_workers or build_config.get("max_workers", 10)
    
    # Load API config and init client
    base_url, api_key, model_name = load_api_config(global_config)
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # Load input raw data
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Get cases from input format
    if isinstance(input_data, dict):
        if "error_cases" in input_data:
            cases = input_data["error_cases"]
            metadata = input_data.get("metadata", {})
        else:
            logger.error("Input data format error: expected dict with 'error_cases' key")
            sys.exit(1)
    else:
        logger.error("Input data format error: expected dict with 'error_cases'")
        sys.exit(1)
    
    logger.info(f"Loaded {len(cases)} raw error cases from {input_path}")
    
    # Handle resume
    processed_indices = set()
    results = []
    
    resume_from = args.resume_from
    if not resume_from and os.path.exists(output_path):
        # Resume from default output if it exists
        resume_from = output_path
    
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from {resume_from}")
        with open(resume_from, 'r', encoding='utf-8') as f:
            resume_data = json.load(f)
        if "cases" in resume_data:
            for processed in resume_data["cases"]:
                original_idx = processed.get("original_index", None)
                if original_idx is not None:
                    processed_indices.add(original_idx)
                    results.append(processed)
        logger.info(f"Resumed {len(results)} already processed cases")
    
    # Filter unprocessed cases
    unprocessed = []
    for idx, case in enumerate(cases):
        if idx not in processed_indices:
            case["original_index"] = idx
            unprocessed.append(case)
    
    # Limit samples for testing if requested
    if args.max_samples is not None and args.max_samples > 0:
        unprocessed = unprocessed[:args.max_samples]
        logger.info(f"Limited to first {args.max_samples} samples for testing")
    
    logger.info(f"{len(unprocessed)} cases remaining to process")
    
    if not unprocessed:
        logger.info("All cases already processed, exiting")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process in parallel
    completed = 0
    total = len(unprocessed)
    
    pbar = tqdm(total=total) if has_tqdm else None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for case in unprocessed:
            future = executor.submit(
                extract_hierarchical_structure,
                client,
                model_name,
                case
            )
            futures.append(future)
        
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            if result:
                # Add original index for resume
                result["original_index"] = result["L0"]["original_index"] if "original_index" in result["L0"] else None
                results.append(result)
            else:
                logger.warning(f"Failed to process case {completed}/{total}")
            
            if pbar:
                pbar.update(1)
            
            # Periodic save every 10 cases
            if completed % 10 == 0:
                logger.info(f"Processed {completed}/{total}, saving intermediate...")
                output_metadata = metadata.copy()
                output_metadata.update({
                    "total_raw_cases": len(cases),
                    "processed_cases": len(results),
                    "failed_cases": completed - len(results),
                    "generation_model": model_name
                })
                output_data = {
                    "metadata": output_metadata,
                    "cases": results
                }
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    if pbar:
        pbar.close()
    
    # Final save
    output_metadata = metadata.copy()
    output_metadata.update({
        "total_raw_cases": len(cases),
        "processed_cases": len(results),
        "failed_cases": len(unprocessed) - len(results),
        "generation_model": model_name
    })
    output_data = {
        "metadata": output_metadata,
        "cases": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Done! Saved {len(results)} processed cases to {output_path}")
    failed = len(unprocessed) - len(results)
    if failed > 0:
        logger.warning(f"{failed} cases failed to process, you can re-run with --resume-from to retry")


if __name__ == "__main__":
    main()
