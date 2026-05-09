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
from .article_matcher import ArticleMatcher

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent

RAG_PREDICTION_PROMPT = """你是一位专业的中国法官，请你根据以下案件事实，结合参考历史案例，判决被告人的罪名、相关法条和刑期。

以下提供**历史案例**作为参考（如果为空，就是没有，忽略即可）：
- 每个案例包含案件关键事实概括、正确的判决推理，以及该案例曾犯过的错误。
- 请你**认真参考其正确的推理方法**。
- 注意：历史案例中的"易错点"仅适用于该案例本身，不代表该罪名/法条/刑期是错误的。请结合当前案件事实独立判断，勿因案例的误判记录而排斥某个罪名。
- 如果某些案例与当前案件不相关，你可以忽略，不必强行使用。

**重要要求：**
1. 输出标准的中国刑法罪名名称，不要编造罪名。被告人可以犯**一罪或数罪**。
2. **关于案件事实中的罪名提示**：CAIL2018数据中，案件事实里有时会出现罪名（如"被告人涉嫌盗窃罪"），这是公诉方指控的罪名，**不是法院最终判决结论**，你必须根据事实独立判断，不能直接采信。
3. 请严格按照JSON格式输出，只输出以下字段：
   - "罪名": 你判决的罪名名称**数组**，如果是一罪就是长度为1的数组
   - "法条": 相关法条编号**数组**（字符串格式）
   - "刑期": 一个对象，包含imprisonment（有期徒刑月份数，无期徒刑或死刑填0）、death_penalty（是否死刑）、life_imprisonment（是否无期徒刑）
   - "罚金": 罚金金额（整数，无罚金填0）
   - "推理过程": 你判决的详细推理过程

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
        logger.info(f"Loaded {len(self.accu)} candidate accusations from {self.accu_path}")
        self.charge_matcher = ChargeMatcher(str(self.accu_path))
        self.article_matcher = ArticleMatcher(str(self.law_path), charge_article_data=str(ROOT_DIR / "data/charge_article_mapping.json"))

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
            rule = L2.get("rule", "")
            reasoning = L2.get("reasoning", "")
            error_reason = L2.get("error_reason", "")

            if case_summary:
                parts.append(f"**案件关键事实概括**：{case_summary}")
            if rule:
                parts.append(f"**判案规则**：{rule}")
            if reasoning:
                parts.append(f"**涵摄分析**：{reasoning}")
            if error_reason:
                parts.append(f"**易错点**：{error_reason}")

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

        if retrieved_cases:
            logger.info("=== Retrieved historical cases ===")
            for i, case in enumerate(retrieved_cases, 1):
                L0 = case.get("L0", {})
                L2 = case.get("L2", {})
                tc = L0.get("true_charges", [])
                pc = L0.get("predicted_charges", [])
                logger.info(f"Case {i}: true={tc}, pred={pc}, L2 keys={list(L2.keys())}")
                logger.info(f"  case_summary: {L2.get('case_summary', '')}")
                logger.info(f"  rule: {L2.get('rule', '')}")
                logger.info(f"  reasoning: {L2.get('reasoning', '')}")
                logger.info(f"  error_reason: {L2.get('error_reason', '')}")

        prompt = RAG_PREDICTION_PROMPT.format(
            fact=fact,
            retrieved_cases=formatted_cases,
        )

        # --- Initial prediction ---
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
        )

        total_prompt_tokens = response.usage.prompt_tokens
        total_completion_tokens = response.usage.completion_tokens

        def _parse_response(resp_text: str) -> dict:
            cleaned = resp_text.strip().removeprefix("```json").removesuffix("```").strip()
            return json.loads(cleaned)

        content = response.choices[0].message.content.strip()
        try:
            result = _parse_response(content)
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

        # --- Article validation feedback loop ---
        valid_articles, invalid_articles = self.article_matcher.validate(pred_articles)
        attempt = 0
        while invalid_articles and attempt < self.article_matcher.max_iterations:
            attempt += 1
            logger.info(f"Article correction attempt {attempt}: invalid={invalid_articles}")
            feedback = (
                f"以下法条编号不在可选范围内：{'、'.join(invalid_articles)}。\n"
                f"可选法条编号（{', '.join(pred_charges)}相关）：{self.article_matcher.get_articles_for_charges(pred_charges) if attempt == 1 else self.article_matcher.valid_list_text}\n"
                f"请根据案件事实重新选择合适的法条编号，仅输出修正后的JSON：\n"
                f'{{"法条": ["编号1", "编号2"]}}'
            )
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": feedback},
                ],
                temperature=0.0,
            )
            total_prompt_tokens += resp.usage.prompt_tokens
            total_completion_tokens += resp.usage.completion_tokens
            correction = resp.choices[0].message.content.strip()
            try:
                corr_data = _parse_response(correction)
                corr_articles = corr_data.get("法条", [])
                if isinstance(corr_articles, str):
                    corr_articles = [corr_articles]
                valid_articles, invalid_articles = self.article_matcher.validate(corr_articles)
                if valid_articles:
                    pred_articles = valid_articles
            except json.JSONDecodeError:
                logger.warning(f"Article correction parse failed: {correction}")
                break

        if invalid_articles:
            logger.warning(f"Articles still invalid after {attempt} corrections: {invalid_articles}")
        pred_articles = valid_articles or pred_articles
        total_tokens = total_prompt_tokens + total_completion_tokens

        logger.info(f"Prediction done: charges={pred_charges}, articles={pred_articles}, tokens={total_tokens}")

        return {
            "pred_charges": pred_charges,
            "pred_articles": pred_articles,
            "pred_term": pred_term,
            "pred_fine": pred_fine,
            "elements": elements,
            "retrieved_cases": retrieved_cases,
            "prompt": prompt,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
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
    print(f"  Reasoning: {result['pred_reasoning']}")
    print(f"  Total tokens: {result['total_tokens']}")
    print(f"  Extracted elements: {result['elements']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
