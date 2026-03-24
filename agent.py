"""
LJP正负案例智能体 - 核心框架
基于prompt工程，引入正负案例作为上下文进行判决预测

Author: Your Name
Date: 2026-03-23
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import logging
import os
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Case:
    """案件数据结构"""
    fact: str              # 案件事实
    charges: List[str]     # 罪名（可多个）
    articles: List[str]    # 相关法条
    judgment: str          # 判决结果
    is_positive: bool      # 是否为正例（True=正例，False=负例）


@dataclass
class PredictionResult:
    """预测结果"""
    predicted_charges: List[str]
    predicted_articles: List[str]
    predicted_judgment: str
    prompt_tokens: int
    completion_tokens: int


class LJPAgentWithRAG:
    """
    基于正负案例对比+RAG检索的LJP智能体
    自动检索top-k相似正例/负例，注入prompt进行预测
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        pos_retriever,
        neg_retriever,
        k_positive: int = 1,
        k_negative: int = 1,
        charge_names: Optional[List[str]] = None,
        article_names: Optional[List[str]] = None
    ):
        # 排除代理
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.pos_retriever = pos_retriever  # 正例检索器
        self.neg_retriever = neg_retriever  # 负例检索器
        self.k_positive = k_positive
        self.k_negative = k_negative
        self.charge_names = charge_names
        self.article_names = article_names
        
        logger.info(f"LJPAgentWithRAG initialized: model={model_name}, k_pos={k_positive}, k_neg={k_negative}")
    
    def build_prompt(
        self,
        target_case: Case,
        positive_examples: List[Case],
        negative_examples: List[Case],
    ) -> str:
        """
        构建包含正负案例的prompt
        """
        system_prompt = self._default_system_prompt()
        
        prompt_parts = [system_prompt, ""]
        
        # 添加可选罪名列表（如果有）
        if self.charge_names and len(self.charge_names) > 0:
            prompt_parts.append("## 可选罪名列表（必须从这里选择，不能自己编造）")
            prompt_parts.append(f"（共{len(self.charge_names)}个罪名，这里展示前100个）：")
            prompt_parts.append(", ".join(self.charge_names[:100]) + "...")
            prompt_parts.append("")
        
        if self.article_names and len(self.article_names) > 0:
            prompt_parts.append("## 可选法条编号（必须从这里选择，不能自己编造）")
            prompt_parts.append(f"（共{len(self.article_names)}个法条，这里展示前100个）：")
            article_ids = [article for article in self.article_names[:100]]
            prompt_parts.append(", ".join(article_ids) + "...")
            prompt_parts.append("")
        
        # 添加正负案例
        if positive_examples:
            prompt_parts.append("## 正例（相似案件，正确判决参考）")
            for i, case in enumerate(positive_examples, 1):
                prompt_parts.append(self._format_case(i, case))
        
        if negative_examples:
            prompt_parts.append("\n## 负例（相似案件，错误判决警示）")
            for i, case in enumerate(negative_examples, 1):
                prompt_parts.append(self._format_case(i, case))
        
        # 添加目标案件
        prompt_parts.append(f"\n## 目标案件（需要你预测判决）")
        prompt_parts.append(self._format_case(0, target_case, include_answer=False))
        
        # 添加预测要求
        prompt_parts.append("""
请根据上述正负参考案例，结合你的法律知识，对目标案件进行预测，输出格式为JSON，注意格式要求：
- 罪名必须从可选罪名列表中选择，直接输出名称，不要加"罪"字后缀（例如："故意伤害" 不是 "故意伤害罪"）
- 法条必须从可选法条编号中选择，只输出编号即可（例如："234" 不是 "《中华人民共和国刑法》第二百三十四条"）
- 如果有多个罪名或法条，输出多个

输出格式：
{
  "reasoning": "你的推理过程，分析对比正负案例为什么得出这个结论",
  "predicted_charges": ["罪名1", "罪名2"],
  "predicted_articles": ["法条编号"],
  "predicted_judgment": "判决结果描述"
}
""")
        
        return "\n".join(prompt_parts)
    
    def _default_system_prompt(self) -> str:
        return """你是一个专业的法律AI助手，擅长中国刑事案件判决预测。
下面会给你提供一些参考案例，分为正例和负例：
- 正例：与当前案件事实相似，判决结果正确的案例，可以参考其判决思路
- 负例：与当前案件事实相似，但判决结果错误的案例，请你避免犯同样的错误

请仔细对比分析，给出最准确的预测。"""
    
    def _format_case(self, index: int, case: Case, include_answer: bool = True) -> str:
        """格式化单个案例"""
        lines = []
        lines.append(f"### 案例 {index}")
        lines.append(f"**案件事实**: {case.fact[:400]}..." if len(case.fact) > 400 else f"**案件事实**: {case.fact}")
        
        if include_answer:
            if case.is_positive:
                lines.append(f"**正确罪名**: {', '.join(case.charges)}")
                lines.append(f"**正确法条**: {', '.join(case.articles)}")
            else:
                lines.append(f"**错误罪名**: {', '.join(case.charges)}")
                lines.append(f"**错误法条**: {', '.join(case.articles)}")
                lines.append(f"**警示**: 这个案件事实和当前案件相似，但这个判决是错误的，请避免同样的错误")
        
        return "\n".join(lines)
    
    def predict(
        self,
        target_case: Case,
        embedding_model
    ) -> PredictionResult:
        """
        完整预测流程：检索 -> 构建prompt -> 预测
        """
        # 对目标案件编码
        target_embedding = embedding_model.encode(target_case.fact, normalize_embeddings=True)
        
        # 检索正负案例
        # pos_retriever 只存正例，所以取 top-k_positive 作为正例
        pos_result = self.pos_retriever.retrieve(
            target_embedding, 
            k_positive=self.k_positive,
            k_negative=0
        )
        positive_examples = pos_result.positive_examples
        
        # neg_retriever 只存负例，所以取 top-k_negative 作为负例
        neg_result = self.neg_retriever.retrieve(
            target_embedding,
            k_positive=self.k_negative,
            k_negative=0
        )
        negative_examples = neg_result.positive_examples  # 负例库每个case已经标记is_positive=False
        
        prompt = self.build_prompt(target_case, positive_examples, negative_examples)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        usage = response.usage
        
        # 解析JSON
        try:
            # 处理markdown代码块
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            predicted_charges = result.get("predicted_charges", [])
            predicted_articles = result.get("predicted_articles", [])
            predicted_judgment = result.get("predicted_judgment", "")
        except Exception as e:
            logger.warning(f"JSON解析失败，使用原始输出: {e}")
            predicted_charges = []
            predicted_articles = []
            predicted_judgment = content
        
        return PredictionResult(
            predicted_charges=predicted_charges,
            predicted_articles=predicted_articles,
            predicted_judgment=predicted_judgment,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens
        )


class DataLoader:
    """加载CAIL2018等数据集"""
    
    @staticmethod
    def load_cail2018(file_path: str) -> List[dict]:
        """加载CAIL2018格式数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} cases from {file_path}")
        return data
    
    @staticmethod
    def convert_to_case(case_dict: dict, is_positive: bool = True) -> Case:
        """将原始数据转换为Case结构
        支持两种格式：
        - 原始CAIL2018格式：charge/article 在顶层
        - 比赛json格式：charge/article 在meta下
        """
        # 处理两种不同格式
        charges = case_dict.get('charge', [])
        articles = case_dict.get('article', [])
        
        if not charges and 'meta' in case_dict:
            # 比赛格式：标签在meta里
            charges = case_dict['meta'].get('accusation', [])
            articles = case_dict['meta'].get('relevant_articles', [])
        
        return Case(
            fact=case_dict.get('fact', ''),
            charges=charges,
            articles=list(map(str, articles)),
            judgment=case_dict.get('judgment', ''),
            is_positive=is_positive
        )


if __name__ == '__main__':
    # 测试框架
    print("LJP正负案例智能体框架初始化成功")
