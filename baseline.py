"""
LJP基础Baseline - Zero-Shot直接预测
不使用任何检索案例，仅输入案件事实，让大模型直接预测罪名、法条和判决

这是最基础的baseline，后续我们会添加正负案例对比做对比实验
"""

from dataclasses import dataclass
from typing import List, Optional
import json
import logging
from openai import OpenAI

from agent import Case, PredictionResult, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZeroShotLJPBaseline:
    """最基础的Zero-Shot LJP基线"""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str = "ark-code-latest"
    ):
        import os
        # 排除代理，防止SSL问题
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name
        logger.info(f"ZeroShotLJPBaseline initialized: model={model_name}")
    
    def build_prompt(self, target_case: Case) -> str:
        """构建zero-shot prompt"""
        # 如果模型已经加载了标签列表，就添加到prompt中，让模型严格从里面选择
        has_labels = hasattr(self, 'charge_names') and self.charge_names and hasattr(self, 'article_names') and self.article_names
        
        prompt = f"""你是一个专业的法律AI助手，请根据以下案件事实，预测该案的罪名、相关法条和判决结果。

## 案件事实
{target_case.fact}

"""
        if has_labels:
            prompt += f"""## 可选罪名列表（必须从下面选择，不能自己编造）
{", ".join(self.charge_names[:100])}...（共{len(self.charge_names)}个罪名）

## 可选法条编号（必须从下面选择，不能自己编造）
{", ".join(self.article_names[:100])}...（共{len(self.article_names)}个法条）

"""
        
        prompt += """请按照JSON格式输出你的预测，注意格式要求：
- 罪名必须从可选罪名列表中选择，直接输出名称，不要加"罪"字后缀（例如："故意伤害" 不是 "故意伤害罪"）
- 法条必须从可选法条编号中选择，只输出编号即可（例如："234"）
- 如果有多个罪名或法条，输出多个

格式：
{
  "reasoning": "你的分析推理过程",
  "predicted_charges": ["罪名1", "罪名2"],
  "predicted_articles": ["法条编号"],
  "predicted_judgment": "判决结果描述"
}
"""
        return prompt
    
    def predict(self, target_case: Case) -> PredictionResult:
        """进行预测"""
        prompt = self.build_prompt(target_case)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  # 预测任务用低温，减少随机性
        )
        
        content = response.choices[0].message.content
        usage = response.usage
        
        # 尝试解析JSON输出
        try:
            # 处理可能的markdown代码块
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


if __name__ == "__main__":
    print("ZeroShotLJPBaseline - 基础基线就绪")
