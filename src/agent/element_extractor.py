#!/usr/bin/env python3
"""
Independent utility module: Legal element extraction  
Extracts seven key legal elements from case facts for retrieval purposes.

Usage:
  from element_extractor import LegalElementExtractor
  extractor = LegalElementExtractor(config_path="config.json")
  elements = extractor.extract(fact)
"""

import json
import os
import logging
import yaml
from typing import Dict
from openai import OpenAI
from pathlib import Path

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent


ELEMENT_EXTRACTION_PROMPT = """你是一位法律AI助手，请你帮我从以下刑事案件中提取七个**定性法律要素**。

### 提取要求：
- **只提取性质判断，不要提取具体人名、地名、具体数值等个性化信息**
- 只保留和定罪相关的法律性质，过滤噪声信息
- 帮助我们检索"在法律要素上相似"的错例，不是检索事实细节相似
- **关于案件事实中的罪名提示**：CAIL2018数据中，案件事实里有时候会出现罪名（如"被告人涉嫌盗窃罪"），这是公诉方指控的罪名，不是法院最终判决结论，最终结论不以该罪名为准。提取要素时请注意区分。

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


class LegalElementExtractor:
    """
    独立的法律要素提取工具
    从原始案件事实中提取七个关键法律要素，用于后续加权检索
    """
    
    def __init__(
        self,
        config_path: str = None,
    ):
        """
        初始化提取器
        
        Args:
            config_path: 配置文件路径，读取API配置
        """
        if config_path is not None:
            config_path = ROOT_DIR / "config" / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化OpenAI客户端
        self.client = self._init_openai_client()
        self.model_name = self.config["api"]["model_name"]
        logger.info("✅ LegalElementExtractor initialized")
    
    def _init_openai_client(self) -> OpenAI:
        """从配置初始化OpenAI客户端"""
        api_key = self._get_api_key()
        base_url = self.config["api"]["base_url"]
        return OpenAI(api_key=api_key, base_url=base_url)
    
    def _get_api_key(self) -> str:
        """获取API key"""
        api_key_env = self.config["api"]["api_key"]
        if api_key_env == "OPENAI_API_KEY":
            key = os.environ.get("OPENAI_API_KEY")
        elif api_key_env == "DEEPSEEK_API_KEY":
            key = os.environ.get("DEEPSEEK_API_KEY")
        else:
            key = api_key_env
        
        if not key:
            raise ValueError(
                "API key not found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable.\n"
                f"Config expects {api_key_env} from config."
            )
        return key
    
    def extract(self, fact: str) -> Dict[str, str]:
        """
        从案件事实中提取七个法律要素
        
        Args:
            fact: 案件事实文本
            
        Returns:
            Dict[str, str]: 提取的七个要素，键：主体、行为、结果、主观、客体、手段、情节
        """
        prompt = ELEMENT_EXTRACTION_PROMPT.format(fact=fact)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        
        content = response.choices[0].message.content.strip()
        # 清理markdown包装
        content = content.removeprefix("```json").removesuffix("```").strip()
        
        try:
            elements = json.loads(content)
        except json.JSONDecodeError as e:
                logger.error(f"Failed to parse elements JSON: {content}")
                raise e
        
        logger.info(f"✅ Extracted legal elements: {list(elements.keys())}")
        return elements


def main():
    """命令行测试入口"""
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT_DIR / "config" / "config.yaml", help="Config file path")
    parser.add_argument("--fact", required=True, help="Text file with case fact")
    args = parser.parse_args()
    
    with open(args.fact, 'r', encoding='utf-8') as f:
        fact = f.read()
    
    extractor = LegalElementExtractor(args.config)
    elements = extractor.extract(fact)
    
    print("\n" + "="*60)
    print("Extracted Legal Elements:")
    print(json.dumps(elements, indent=2, ensure_ascii=False))
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
