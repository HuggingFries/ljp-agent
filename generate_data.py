"""
错例生成脚本
- 从训练集随机采样N条
- 一半采样直接作为正例（保存原始标签）
- 一半让大模型生成错误罪名+法条，作为错例保存
"""

import os
import json
import random
import argparse
import logging
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

import os
import json
from openai import OpenAI
from agent import Case, DataLoader

def load_api_config(config: dict):
    """加载API配置
    - base_url, model_name 直接从config读
    - api_key 从环境变量读，env name 存在config里
    """
    api_config = config.get("api", {})
    base_url = api_config.get("base_url")
    api_key_env = api_config.get("api_key_env", "DEEPSEEK_API_KEY")
    model_name = api_config.get("model_name")
    
    api_key = os.getenv(api_key_env)
    
    if not all([base_url, api_key, model_name]):
        raise ValueError(
            f"Missing API configuration:\n"
            f"  - base_url: {base_url}\n"
            f"  - api_key from env '{api_key_env}': {api_key is not None}\n"
            f"  - model_name: {model_name}"
        )
    
    return base_url, api_key, model_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_wrong_label(
    client: OpenAI,
    model_name: str,
    fact: str,
    true_charges: List[str],
    true_articles: List[str]
) -> Tuple[List[str], List[str]]:
    """让大模型生成一个错误的罪名和法条"""
    
    prompt = f"""你是一个法律AI助手，请根据给定的案件事实和正确罪名，给出一个**常见的错误判决**，只需要输出错误罪名和错误法条，不需要解释原因。

案件事实:
{fact[:800]}...

正确罪名: {', '.join(true_charges)}
正确法条编号: {', '.join(true_articles)}

请按照JSON格式输出：
{{
  "wrong_charges": ["错误罪名1", "错误罪名2"],
  "wrong_articles": ["错误法条编号"]
}}
"""
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8  # 温度高一点，增加错误多样性
    )
    
    content = response.choices[0].message.content
    # 解析JSON
    try:
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        wrong_charges = result.get("wrong_charges", [])
        wrong_articles = result.get("wrong_articles", [])
        return wrong_charges, wrong_articles
    except Exception as e:
        logger.error(f"Failed to parse: {e}, content: {content}")
        return [], []


def main():
    parser = argparse.ArgumentParser(description='Generate negative examples for LJP')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--train-file', type=str, 
                       default='data/final_all_data/first_stage/train.json',
                       help='CAIL2018 training json path (overrides config)')
    parser.add_argument('--total-sample', type=int, default=None,
                       help='Total number of samples to sample (overrides config)')
    parser.add_argument('--neg-sample', type=int, default=None,
                       help='Number of negative examples to generate (overrides config)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (overrides config)')
    args = parser.parse_args()
    
    # 读取配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 命令行覆盖配置
    eval_config = config.get("evaluation", {})
    train_file = args.train_file or eval_config.get("test_file", "data/final_all_data/first_stage/train.json")
    total_sample = args.total_sample or 500
    neg_sample = args.neg_sample or 250
    seed = args.seed or eval_config.get("seed", 42)
    
    random.seed(seed)
    
    # 加载API配置
    base_url, api_key, model_name = load_api_config(config)
    
    # 排除代理
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # 加载训练数据
    train_data = DataLoader.load_cail2018(args.train_file)
    logger.info(f"Loaded {len(train_data)} training samples")
    
    # 随机采样
    all_samples = random.sample(train_data, args.total_sample)
    logger.info(f"Sampled {args.total_sample} total samples")
    
    # 拆分：正样本和负样本
    pos_samples = all_samples[:args.total_sample - args.neg_sample]
    neg_samples_input = all_samples[args.total_sample - args.neg_sample:]
    logger.info(f"Positive samples: {len(pos_samples)}, Negative samples to generate: {len(neg_samples_input)}")
    
    # 保存正样本（原始标签）
    pos_output = []
    for item in pos_samples:
        case = DataLoader.convert_to_case(item)
        pos_output.append({
            "fact": case.fact,
            "charges": case.charges,
            "articles": case.articles
        })
    
    with open("data/sampled_positives.json", 'w', encoding='utf-8') as f:
        json.dump(pos_output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(pos_output)} positive samples to data/sampled_positives.json")
    
    # 生成负样本
    neg_output = []
    for i, item in enumerate(neg_samples_input):
        case = DataLoader.convert_to_case(item)
        true_charges = case.charges
        true_articles = case.articles
        
        logger.info(f"Generating negative {i+1}/{len(neg_samples_input)}")
        
        wrong_charges, wrong_articles = generate_wrong_label(
            client, model_name, case.fact, true_charges, true_articles
        )
        
        if wrong_charges:
            neg_output.append({
                "fact": case.fact,
                "true_charges": true_charges,
                "true_articles": true_articles,
                "wrong_charges": wrong_charges,
                "wrong_articles": wrong_articles
            })
        
        # 每生成10个保存一次，防止中途崩溃
        if (i + 1) % 10 == 0:
            with open("data/generated_negatives.json.tmp", 'w', encoding='utf-8') as f:
                json.dump(neg_output, f, indent=2, ensure_ascii=False)
    
    # 保存最终负样本
    with open("data/generated_negatives.json", 'w', encoding='utf-8') as f:
        json.dump(neg_output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(neg_output)} generated negative examples to data/generated_negatives.json")
    logger.info("Done!")


if __name__ == "__main__":
    main()
