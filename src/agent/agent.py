#!/usr/bin/env python3
"""
RAG Agent for LJP with unified historical case retrieval and sentence prediction.

Workflow:
1. Extract legal elements from input fact
2. Retrieve similar historical cases from unified KB
3. Inject case info (both correct reasoning + error analysis) into prompt
4. Run final prediction (charges + articles + term + fine) with LLM

Usage:
    from agent import LJPRAGAgent
    agent = LJPRAGAgent(config_path="config.yaml")
    prediction = agent.predict(fact_text)
"""

import json
import yaml
import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pathlib import Path
from .element_extractor import LegalElementExtractor
from .retriever import LJPRetriever
from .charge_matcher import ChargeMatcher

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent

RAG_PREDICTION_PROMPT = """你是一位专业的中国法官，请你根据以下案件事实，结合参考历史案例，判决被告人的罪名、相关法条和刑期。

以下提供**历史案例**作为参考（如果为空，就是没有，忽略即可）：
- 每个案例包含案件关键事实概括、正确的判决推理（为什么该这么判），以及曾经犯过的错误分析（为什么不该那么判）。
- 请你**认真参考其正确的推理方法**，并**吸取其错误教训，避免重蹈覆辙**。
- 如果某些案例与当前案件不相关，你可以忽略，不必强行使用。

**重要要求：**
1. 输出标准的中国刑法罪名名称，不要编造罪名。被告人可以犯**一罪或数罪**，不允许输出列表外的罪名。
2. 你只能从下面给定的法条编号中选择相关法条，可以选多个。
3. **关于案件事实中的罪名提示**：CAIL2018数据中，案件事实里有时会出现罪名（如"被告人涉嫌盗窃罪"），这是公诉方指控的罪名，**不是法院最终判决结论**，你必须根据事实独立判断，不能直接采信。
4. 请严格按照JSON格式输出，只输出以下字段：
   - "罪名": 你判决的罪名名称**数组**，如果是一罪就是长度为1的数组
   - "法条": 相关法条编号**数组**（字符串格式）
   - "刑期": 一个对象，包含imprisonment（有期徒刑月份数，无期徒刑或死刑填0）、death_penalty（是否死刑）、life_imprisonment（是否无期徒刑）
   - "罚金": 罚金金额（整数，无罚金填0）
   - "推理过程": 你判决的详细推理过程

相关法条编号：
{laws}

案件事实：
{fact}

### 历史案例（供参考，如果为空就没有）：
{retrieved_cases}

请输出JSON：
"""


class LJPRAGAgent:
    """
    Main RAG Agent for Legal Judgment Prediction with unified historical cases.
    Predicts charges, articles, sentence term, and fine.
    """

    def __init__(
        self,
        config_path: str = None,
        device: str = "cpu",
    ):
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

        self.element_extractor = LegalElementExtractor(config_path=config_path)
        logger.info("LegalElementExtractor initialized")

        self.retriever = LJPRetriever(
            config_path=config_path,
            device=device,
        )

        logger.info("LJPRAGAgent initialized successfully")

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
                "API key not found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable or paste the key into the config file.\n"
                f"Config expects {api_key_env} from config."
            )
        return key

    def extract_legal_elements(self, fact: str) -> Dict[str, str]:
        return self.element_extractor.extract(fact)

    def format_retrieved_cases(self, retrieved_cases: List[Dict[str, Any]]) -> str:
        """
        Format retrieved historical cases for prompt injection.
        Each case contains both correct reasoning and error analysis.
        """
        if not retrieved_cases:
            return ""

        parts = [""]
        for i, case in enumerate(retrieved_cases, 1):
            L0 = case.get("L0", {})
            L2 = case.get("L2", {})

            parts.append(f"### 历史案例 {i}:")

            fact = L0.get("fact", "").strip()
            true_charges = ";".join(L0.get("true_charges", []))
            true_articles = ";".join(L0.get("true_articles", []))
            pred_charges = ";".join(L0.get("predicted_charges", []))
            pred_articles = ";".join(L0.get("predicted_articles", []))

            case_summary = L2.get("case_summary", "")
            correct_reasoning = L2.get("correct_reasoning", "")
            wrong_reasoning = L2.get("wrong_reasoning", "")

            if case_summary:
                parts.append(f"**案件关键事实概括**：{case_summary}")
            if correct_reasoning:
                parts.append(f"**正确判决的原因**：{correct_reasoning}")
            if wrong_reasoning:
                parts.append(f"**错误判决（模型曾误判为{'+'.join(pred_charges) if pred_charges else '无'}）的原因**：{wrong_reasoning}")

        return "\n".join(parts)

    def predict(
        self,
        fact: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Run full RAG prediction pipeline with unified historical cases.

        Args:
            fact: Input case fact text
            top_k: Number of historical cases to retrieve

        Returns:
            Dict with prediction result including charges, articles, term, fine
        """
        elements = self.element_extractor.extract(fact)

        retrieved_cases = self.retriever.retrieve(fact, elements, top_k)
        formatted_cases = self.format_retrieved_cases(retrieved_cases)

        law_text = "\n".join([f"- {law}" for law in self.law])
        prompt = RAG_PREDICTION_PROMPT.format(
            fact=fact,
            retrieved_cases=formatted_cases,
            laws=law_text,
        )

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
            pred_reasoning = result.get("推理过程", "")

            pred_term = result.get("刑期", {})
            if isinstance(pred_term, str):
                pred_term = {}
            pred_fine = result.get("罚金", 0)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse prediction JSON: {content}")
            pred_charges = []
            pred_articles = []
            pred_reasoning = ""
            pred_term = {}
            pred_fine = 0

        logger.info(f"Prediction done: charges={pred_charges}, articles={pred_articles}, tokens={total_tokens}")

        return {
            "pred_charges": pred_charges,
            "pred_articles": pred_articles,
            "pred_term": pred_term,
            "pred_fine": pred_fine,
            "elements": elements,
            "retrieved_cases": retrieved_cases,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "pred_reasoning": pred_reasoning,
        }


def main():
    """Quick test for agent."""
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT_DIR / "config" / "config.yaml", help="Config file path")
    parser.add_argument("--fact", help="Input fact text file", default=ROOT_DIR / "data" / "sample.txt")
    parser.add_argument("--top-k", type=int, default=3, help="Number of cases to retrieve")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()

    with open(args.fact, 'r', encoding='utf-8') as f:
        fact = f.read()

    agent = LJPRAGAgent(config_path=args.config, device=args.device)
    result = agent.predict(fact, top_k=args.top_k)

    print("\n" + "="*50)
    print("Prediction Result:")
    print(f"  Charges: {result['pred_charges']}")
    print(f"  Articles: {result['pred_articles']}")
    print(f"  Term: {result['pred_term']}")
    print(f"  Fine: {result['pred_fine']}")
    print(f"  Total tokens: {result['total_tokens']}")
    print(f"  Extracted elements: {result['elements']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
