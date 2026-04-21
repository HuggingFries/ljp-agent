#!/usr/bin/env python3
"""
RAG Agent main logic for LJP negative example enhancement.
Extracts legal elements from input case, retrieves similar negative cases,
injects error information into prompt, and runs final prediction.

Usage:
1. Import and use in code:
    from agent import LJPRAGAgent
    agent = LJPRAGAgent(config_path="config.json")
    prediction = agent.predict(fact_text)

2. Run as script for quick test:
    python agent.py [options]

    [options]
    
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from element_extractor import LegalElementExtractor
from retriever import LJPRetriever

logger = logging.getLogger(__name__)


# Element extraction is now handled by independent element_extractor.py
# Prompt kept here for reference, code moved to element_extractor


RAG_PREDICTION_PROMPT = """你是一位专业的中国法官，请你根据以下案件事实，结合参考错例，判决被告人的罪名和相关法条。
以下提供的是**过往模型在类似案件中被判错的真实案例**，它们与当前案件事实相似，请你仔细阅读并吸取前车之鉴，避免重蹈覆辙。

**重要要求：**
1. 你只能从下面给定的候选罪名列表中选择，被告人可以犯**一罪或数罪**，不允许输出列表外的罪名
2. 你只能从下面给定的法条编号中选择相关法条，可以选多个
3. 请严格按照JSON格式输出，只输出两个字段：
   - "罪名": 你判决的罪名名称**数组**，如果是一罪就是长度为1的数组
   - "法条": 相关法条编号**数组**（字符串格式）

候选罪名列表：
{accusations}

相关法条编号：
{laws}

案件事实：
{fact}

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
        config_path: str = "config.json",
        device: str = "cpu",
    ):
        """
        Initialize agent from config.
        
        Args:
            config_path: Path to config json file
            device: Device for embedding model in retriever
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Load accusation list and law list
        self.accusations = self._load_label_file("data/accu.txt")
        self.laws = self._load_label_file("data/law.txt")
        logger.info(f"Loaded {len(self.accusations)} candidate accusations from data/accu.txt")
        logger.info(f"Loaded {len(self.laws)} candidate laws from data/law.txt")
        
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
        
        logger.info("✅ LJPRAGAgent initialized successfully")
    
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
                "API key not found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable.\n"
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
    
    def format_negative_info(self, retrieved_negatives: List[Dict[str, Any]]) -> str:
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
        
        parts = ["参考错例：以下是与本案相似、但曾被错误判决的案例，请认真吸取教训，避免犯同样的错误：\n"]
        for i, case in enumerate(retrieved_negatives, 1):
            L0 = case.get("L0", {})
            
            fact = L0.get("fact", "")
            error_reason = L0.get("error_reason", "")
            predicted_charge = L0.get("predicted_charge", "")
            true_charge = L0.get("true_charge", "")
            
            parts.append(f"### 错例 {i}:\n")
            if fact:
                parts.append(f"**案件事实**：\n{fact}\n")
            if predicted_charge:
                parts.append(f"- **错误判决**：{predicted_charge}")
            if true_charge:
                parts.append(f"- **正确判决**：{true_charge}")
            if error_reason:
                parts.append(f"- **错误原因**：{error_reason}")
            parts.append("")
        
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
        formatted_negatives = self.format_negative_info(retrieved_negatives)
        
        # Step 4: Run final prediction
        # Join accusations and laws into strings
        accu_text = "\n".join([f"- {accu}" for accu in self.accusations])
        law_text = "\n".join([f"- {law}" for law in self.laws])
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
        except json.JSONDecodeError as e:
                logger.error(f"Failed to parse prediction JSON: {content}")
                pred_charges = []
                pred_articles = []
        
        logger.info(f"✅ Prediction done: charges={pred_charges}, articles={pred_articles}, tokens={total_tokens}")
        
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
        }


def main():
    """Quick test for agent."""
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--fact", help="Input fact text file", required=True)
    parser.add_argument("--top-k", type=int, default=3, help="Number of negative cases")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()
    
    with open(args.fact, 'r', encoding='utf-8') as f:
        fact = f.read()
    
    agent = LJPRAGAgent(config_path=args.config, device=args.device)
    result = agent.predict(fact, top_k=args.top_k)
    
    print("\n" + "="*50)
    print("Prediction Result:")
    print(f"  Prediction: {result['pred_charge']}")
    print(f"  Articles: {result['pred_articles']}")
    print(f"  Total tokens: {result['total_tokens']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
