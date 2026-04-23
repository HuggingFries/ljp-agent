#!/usr/bin/env python3
"""
RAG Agent main logic for LJP negative example enhancement.
Extracts legal elements from input case, retrieves similar negative cases,
injects error information into prompt, and runs final prediction.

Workflow: xxx

Usage:
1. Import and use in code:
    from agent import LJPRAGAgent
    agent = LJPRAGAgent(config_path="config.json")
    prediction = agent.predict(fact_text)

2. Run as script for quick test:
    python src/agent/agent.py [options]

    [options]
        --config CONFIG_PATH   Path to config yaml file (default: config/config.yaml)
        --fact FACT_PATH       Path to input fact text file (default: data/sample.txt)
        --top-k TOP_K          Number of negative cases to retrieve (default: 3)
        --device DEVICE        Device for embedding model (default: cpu)
    
"""

import json
import yaml
import os
import logging
from tkinter import CURRENT
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pathlib import Path
from .element_extractor import LegalElementExtractor
from .retriever import LJPRetriever

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent

RAG_PREDICTION_PROMPT = """你是一位专业的中国法官，请你根据以下案件事实，结合参考错例，判决被告人的罪名和相关法条。
以下提供的是**过往模型在类似案件中被判错的真实案例**，它们与当前案件事实相似，请你仔细阅读并吸取前车之鉴，避免重蹈覆辙。当然，如果你认为没有参考价值，可以不必强行使用它们。

**重要要求：**
1. 你只能从下面给定的候选罪名列表中选择，被告人可以犯**一罪或数罪**，不允许输出列表外的罪名
2. 你只能从下面给定的法条编号中选择相关法条，可以选多个
3. **关于案件事实中的罪名提示**：CAIL2018数据中，案件事实里有时候会出现罪名（如"被告人涉嫌盗窃罪"），这是公诉方指控的罪名，不是法院最终判决结论，你不要直接采信该罪名，需要根据事实独立判断。
4. 请严格按照JSON格式输出，只输出三个字段：
   - "罪名": 你判决的罪名名称**数组**，如果是一罪就是长度为1的数组
   - "法条": 相关法条编号**数组**（字符串格式）
   - "推理过程": 你判决的详细推理过程，包括根据事实进行判案的推理，和针对参考错例的教训的推理过程

候选罪名列表：
{accusations}

相关法条编号：
{laws}

案件事实：
{fact}

检索到的历史上被判错的参考案例信息：
{retrieved_negatives}

请输出JSON：
"""


class LJPRAGAgent:
    """
    Main RAG Agent for Legal Judgment Prediction with negative example enhancement.
    Workflow:
    1. Extract 7 legal elements from input fact (using independent element_extractor)
    2. Retrieve similar negative error cases
    3. Inject error information into prompt
    4. Run final prediction with DeepSeek API
    """
    
    def __init__(
        self,
        config_path: str = None,
        device: str = "cpu",
    ):
        """
        Initialize agent from config.
        
        Args:
            config_path: Path to config json file
            device: Device for embedding model in retriever
        """
        if config_path is not None:
            config_path = ROOT_DIR / "config" / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Load accusation list and law list
        self.accu_path = ROOT_DIR / self.config["data"]["accu_path"]
        self.law_path = ROOT_DIR / self.config["data"]["law_path"]
        self.accu = self._load_label_file(self.accu_path)
        self.law = self._load_label_file(self.law_path)
        logger.info(f"Loaded {len(self.accu)} candidate accusations from {self.accu_path}")
        logger.info(f"Loaded {len(self.law)} candidate laws from {self.law_path}")
        
        # Initialize OpenAI client (DeepSeek compatible)
        api_key = self._get_api_key()
        base_url = self.config["api"]["base_url"]
        self.model_name = self.config["api"]["model_name"]
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"Initialized OpenAI client at {base_url}, model: {self.model_name}")
        
        # Initialize independent legal element extractor tool
        self.element_extractor = LegalElementExtractor(config_path=config_path)
        logger.info("LegalElementExtractor initialized")
        
        # Initialize retriever (we handle extraction, pass elements to it)
        self.retriever = LJPRetriever(
            config_path=config_path,
            device=device,
        )
        
        logger.info("Done! LJPRAGAgent initialized successfully")
    
    def _load_label_file(self, path: str) -> List[str]:
        """Load label list from text file (one label per line)."""
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    
    def _get_api_key(self) -> str:
        """
        Get API key from environment:
        - Prefer OPENAI_API_KEY
        - Fallback to DEEPSEEK_API_KEY
        """
        api_key_env = self.config["api"]["api_key"]
        if api_key_env == "OPENAI_API_KEY":
            key = os.environ.get("OPENAI_API_KEY")
        elif api_key_env == "DEEPSEEK_API_KEY":
            key = os.environ.get("DEEPSEEK_API_KEY")
        else:
            key = api_key_env  # Directly use the value from config
        
        if not key:
            raise ValueError(
                "API key not found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable or paste the key into the config file.\n"
                f"Config expects {api_key_env} from config."
            )
        return key
    
    def extract_legal_elements(self, fact: str) -> Dict[str, str]:
        """
        Extract 7 legal elements from input fact using LLM.
        Now delegated to independent element_extractor tool.
        
        Args:
            fact: Input case fact text
        
        Returns:
            Dict with 7 legal elements
        """
        return self.element_extractor.extract(fact)
    
    def format_negative_info(self, retrieved_negatives: List[Dict[str, Any]], layer="L2L3") -> str:
        """
        Format retrieved negative cases for prompt injection.
        Inject L0 (full original case) layer.
        
        Args:
            retrieved_negatives: Output from retriever.retrieve_negative
        
        Returns:
            Formatted string for prompt
        """
        if not retrieved_negatives:
            return ""
        
        parts = ["参考错例：以下是与本案相似、但曾被错误判决的案例，请认真吸取教训，避免犯同样的错误（如果某些案件不具有参考性，可以忽略，不必强行参考）：\n"]
        for i, case in enumerate(retrieved_negatives, 1):
            L0 = case.get("L0", {}) # original info
            L1 = case.get("L1", {}) # extracted elements
            L2 = case.get("L2", {}) # error analysis 1
            L3 = case.get("L3", {}) # error analysis 2

            parts.append(f"### 错例 {i}:\n")
            
            # L0 layer
            fact = L0.get("fact", "")
            pred_charge_list = L0.get("predicted_charges", [])
            true_charge_list = L0.get("true_charges", [])
            pred_article_list = L0.get("predicted_articles", [])
            true_article_list = L0.get("true_articles", [])
            pred_reasoning = L0.get("pred_reasoning", "")
            # L1 layer
            legal_elements = L1.get("legal_elements", {})
            # L2 layer
            case_summary = L2.get("case_summary", "")
            correct_reasoning = L2.get("correct_reasoning", "")
            wrong_reasoning = L2.get("wrong_reasoning", "")
            error_summary = L2.get("error_summary", [])
            controversy_score = L2.get("controversy_score", 1.0)
            # L3 layer(not used for now, added later)
            experience = L3.get("experience", "")
            lesson = L3.get("lesson", "")
            hint = L3.get("hint", "")

            pred_charges = ";".join(pred_charge_list)
            true_charges = ";".join(true_charge_list)
            pred_articles = ";".join(pred_article_list)
            true_articles = ";".join(true_article_list)
            
            if 'L0' in layer:
                parts.append("以下是该错例的元信息")
                parts.append(f"**案件事实**：\n{fact}\n")

                parts.append(f"- **错误判决**：{pred_charges}")
                parts.append(f"- **正确判决**：{true_charges}")
                parts.append(f"- **错误相关法条**：{pred_articles}")
                parts.append(f"- **正确相关法条**：{true_articles}")

                parts.append(f"- **判决原因**：{pred_reasoning}")

            if 'L1' in layer:
                parts.append("以下是该错例案件事实的七要素：")
                parts.append(f"**案件七要素**：")
                for name, value in legal_elements.items():
                    parts.append(f"  - {name}：{value}")

            if 'L2' in layer:
                parts.append("以下是针对该错例的具体分析：")
                parts.append(f"**案件关键事实概括**：{case_summary}")
                parts.append(f"**应该判处{true_charges}和引用{true_articles}的原因**：{correct_reasoning}")
                parts.append(f"**不该判处{pred_charges}和引用{pred_articles}的原因**：{wrong_reasoning}")
                parts.append(f"**错误总结**：{'；'.join(error_summary)}")
                parts.append(f"**该案例的争议度评分（0-1）分，分数越低表示该案例越具有参考价值，越高表示争议越大）**：{controversy_score}")
            
            if 'L3' in layer:
                parts.append("以下是从该错例中提取的较为普适的经验/教训/提示")
                parts.append(f"**经验教训总结**：{experience}")
                parts.append(f"**对本案的启示**：{lesson}")
                parts.append(f"**给法官的提示**：{hint}")

        return "\n".join(parts)
    
    def predict(
        self,
        fact: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Run full RAG prediction pipeline.
        
        Workflow per your design:
        1. agent calls element_extractor to extract elements
        2. agent passes elements + fact to retriever
        3. get retrieved results from retriever
        4. inject into prompt and run final prediction
        
        Args:
            fact: Input case fact text
            top_k: Number of negative cases to retrieve
        
        Returns:
            Dict with prediction result and all intermediate steps
        """
        # Step 1: Extract legal elements (agent calls independent tool)
        elements = self.element_extractor.extract(fact)
        
        # Step 2: Retrieve similar negative cases (agent passes extracted elements to retriever)
        retrieved_negatives = self.retriever.retrieve_negative(fact, elements, top_k)
        
        # Step 3: Format negative information for prompt
        formatted_negatives = self.format_negative_info(retrieved_negatives, layer="L2L3")  # You can choose which layers to include in the prompt
        
        # Step 4: Run final prediction
        # Join accusations and laws into strings
        accu_text = "\n".join([f"- {accu}" for accu in self.accu])
        law_text = "\n".join([f"- {law}" for law in self.law])
        prompt = RAG_PREDICTION_PROMPT.format(
            fact=fact,
            retrieved_negatives=formatted_negatives,
            accusations=accu_text,
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
        # Clean up markdown wrapping
        content = content.removeprefix("```json").removesuffix("```").strip()
        
        # Parse JSON output
        try:
            result = json.loads(content)
            pred_charges = result.get("罪名", [])
            if isinstance(pred_charges, str):
                pred_charges = [pred_charges]
            pred_articles = result.get("法条", [])
            if isinstance(pred_articles, str):
                pred_articles = [pred_articles]
            # Remove trailing "罪" from each charge to match label format
            pred_charges = [c.strip().removesuffix("罪") for c in pred_charges]
            pred_reasoning = result.get("推理过程", "")
        except json.JSONDecodeError as e:
                logger.error(f"Failed to parse prediction JSON: {content}")
                pred_charges = []
                pred_articles = []
                pred_reasoning = ""
        
        logger.info(f"Prediction done: charges={pred_charges}, articles={pred_articles}, tokens={total_tokens}")
        
        return {
            "prediction": pred_charges[0] if len(pred_charges) > 0 else "",  # For backward compatibility
            "pred_charges": pred_charges,
            "pred_articles": pred_articles,
            "elements": elements,
            "retrieved_negatives": retrieved_negatives,
            "prompt": prompt,  # Return the full prompt for inspection
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
    parser.add_argument("--top-k", type=int, default=3, help="Number of negative cases")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()
    
    with open(args.fact, 'r', encoding='utf-8') as f:
        fact = f.read()
    
    agent = LJPRAGAgent(config_path=args.config, device=args.device)
    result = agent.predict(fact, top_k=args.top_k)
    
    print("\n" + "="*50)
    print("Prediction Result:")
    print(f"  Prediction: {result['pred_charges']}")
    print(f"  Articles: {result['pred_articles']}")
    print(f"  Total tokens: {result['total_tokens']}")
    print(f"  Extracted elements: {result['elements']}")
    print(f"  Retrieved negatives: {result['retrieved_negatives']}")
    print(f"  Prompt:\n{result['prompt']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
