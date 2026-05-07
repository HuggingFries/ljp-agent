#!/usr/bin/env python3
"""
Pure LLM baseline for LJP, no RAG enhancement.
Same API interface as LJPRAGAgent for direct comparison.
Now includes sentence (term + fine) prediction.

Usage:
    from baseline import LJPBaseline
    baseline = LJPBaseline(config_path="config.yaml")
    prediction = baseline.predict(fact_text)
"""

import json
import yaml
import os
import logging
from typing import Dict, Any, List
from openai import OpenAI
from pathlib import Path
from src.agent.charge_matcher import ChargeMatcher

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent

BASELINE_PROMPT = """你是一位专业的中国法官，请你根据以下案件事实，判决被告人的罪名、相关法条和刑期。

**重要要求：**
1. 输出标准的中国刑法罪名名称，不要编造罪名。被告人可以犯**一罪或数罪**，不允许输出列表外的罪名
2. 你只能从下面给定的法条编号中选择相关法条，可以选多个
3. **关于案件事实中的罪名提示**：CAIL2018数据中，案件事实里有时候会出现罪名（如"被告人涉嫌盗窃罪"），这是公诉方指控的罪名，不是法院最终判决结论，你不要直接采信该罪名，需要根据事实独立判断。
4. 请严格按照JSON格式输出，只输出以下字段：
   - "罪名": 你判决的罪名名称**数组**，如果是一罪就是长度为1的数组
   - "法条": 相关法条编号**数组**（字符串格式）
   - "刑期": 一个对象，包含imprisonment（有期徒刑月份数，无期徒刑或死刑填0）、death_penalty（是否死刑）、life_imprisonment（是否无期徒刑）
   - "罚金": 罚金金额（整数，无罚金填0）

相关法条编号：
{laws}

案件事实：
{fact}

请输出JSON：
"""


class LJPBaseline:
    """
    Pure LLM baseline for Legal Judgment Prediction, no RAG.
    Predicts charges, articles, sentence term, and fine.
    """

    def __init__(self, config_path: str = None):
        if config_path is not None:
            config_path = ROOT_DIR / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.accu_path = ROOT_DIR / self.config["data"]["accu_path"]
        self.law_path = ROOT_DIR / self.config["data"]["law_path"]
        self.accu = self._load_label_file(self.accu_path)
        self.law = self._load_label_file(self.law_path)
        logger.info(f"Loaded {len(self.accu)} candidate accusations from {self.accu_path}")
        logger.info(f"Loaded {len(self.law)} candidate laws from {self.law_path}")
        self.charge_matcher = ChargeMatcher(str(self.accu_path))

        api_key = self._get_api_key()
        base_url = self.config["api"]["base_url"]
        self.model_name = self.config["api"]["model_name"]

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"Initialized OpenAI client at {base_url}, model: {self.model_name}")
        logger.info("LJPBaseline initialized successfully")

    def _load_label_file(self, path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

    def _get_api_key(self) -> str:
        api_key_env = self.config["api"]["api_key"]
        if api_key_env == "OPENAI_API_KEY":
            key = os.environ.get("OPENAI_API_KEY")
        elif api_key_env == "DEEPSEEK_API_KEY":
            key = os.environ.get("DEEPSEEK_API_KEY")
        else:
            key = api_key_env

        if not key:
            raise ValueError(
                "API key not found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable or paste your API key in the config/config.yaml.\n"
                f"Config expects {api_key_env} from config."
            )
        return key

    def predict(self, fact: str) -> Dict[str, Any]:
        """
        Run baseline prediction directly from raw fact.
        Predicts charges, articles, term, and fine.

        Args:
            fact: Input case fact text

        Returns:
            Dict with prediction result
        """
        law_text = "\n".join([f"- {law}" for law in self.law])
        prompt = BASELINE_PROMPT.format(fact=fact, laws=law_text)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        content = response.choices[0].message.content.strip()
        content = content.removeprefix("```json").removesuffix("```").strip()

        try:
            result = json.loads(content)
            pred_charges = result.get("罪名", [])
            if isinstance(pred_charges, str):
                pred_charges = [pred_charges]
            pred_articles = result.get("法条", [])
            if isinstance(pred_articles, str):
                pred_articles = [pred_articles]
            pred_charges = self.charge_matcher.map_charges(pred_charges)
            pred_term = result.get("刑期", {})
            if isinstance(pred_term, str):
                pred_term = {}
            pred_fine = result.get("罚金", 0)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse prediction JSON: {content}")
            pred_charges = []
            pred_articles = []
            pred_term = {}
            pred_fine = 0

        logger.info(f"Baseline prediction done: charges={pred_charges}, articles={pred_articles}, tokens={total_tokens}")

        return {
            "pred_charges": pred_charges,
            "pred_articles": pred_articles,
            "pred_term": pred_term,
            "pred_fine": pred_fine,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


def main():
    """Quick test for baseline."""
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--fact", default=f"{ROOT_DIR}/data/sample.txt", help="Input fact text file")
    args = parser.parse_args()

    with open(args.fact, 'r', encoding='utf-8') as f:
        fact = f.read()

    if not fact or not fact.strip():
        raise ValueError("Input fact is empty.")

    baseline = LJPBaseline(config_path=args.config)
    result = baseline.predict(fact)

    print("\n" + "="*50)
    print("Input Fact:\n")
    print(fact)
    print("\n" + "="*50)
    print("Baseline Prediction Result:")
    print(f"  Charges: {result['pred_charges']}")
    print(f"  Articles: {result['pred_articles']}")
    print(f"  Term: {result['pred_term']}")
    print(f"  Fine: {result['pred_fine']}")
    print(f"  Total tokens: {result['total_tokens']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
