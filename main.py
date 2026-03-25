"""
LJP正负案例智能体 - 主入口
支持多种模式：
- baseline: zero-shot 直接预测（无案例）
- positive-only: 只使用正例检索
- positive-negative: 正负案例对比（完整方法）

Usage:
  python main.py --mode baseline --input data/sample_case.json
  python main.py --mode positive-negative --input data/sample_case.json
"""

import argparse
import json
import os
import sys

# 自动添加用户site-packages路径 - 跨平台兼容
# Linux/WSL: ~/.local/lib/pythonX.X/site-packages
# Windows: 直接使用conda/当前Python环境，不需要额外加路径
if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
    user_local_base = os.path.expanduser("~/.local/lib")
    if os.path.exists(user_local_base):
        for version_dir in os.listdir(user_local_base):
            if version_dir.startswith("python"):
                user_local_lib = os.path.join(user_local_base, version_dir, "site-packages")
                if user_local_lib not in sys.path:
                    sys.path.insert(0, user_local_lib)
                break

try:
    from dotenv import load_dotenv
except ImportError:
    print("⚠️  找不到模块 dotenv，请先安装依赖：")
    print("   conda create -n ljp-agent python=3.11")
    print("   conda activate ljp-agent")
    print("   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
    sys.exit(1)

from agent import Case, DataLoader
from baseline import ZeroShotLJPBaseline


def load_api_config():
    """加载API配置
    从项目根目录读取配置文件 config.json，统一管理
    格式示例：
    {
      "api_key": "your-api-key",
      "model_name": "deepseek-chat",
      "base_url": "https://api.deepseek.com/v1"
    }
    """
    # 配置文件放在项目根目录
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    
    if not os.path.exists(config_path):
        print(f"⚠️  找不到配置文件 {config_path}")
        print("请在项目根目录创建 config.json，格式如下：")
        print("""{
  "api_key": "your-deepseek-api-key",
  "model_name": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1"
}""")
        sys.exit(1)
    
    # 读取并解析配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    api_key = config.get("api_key")
    model_name = config.get("model_name", "deepseek-chat")
    base_url = config.get("base_url", "https://api.deepseek.com/v1")
    
    if not api_key:
        print(f"⚠️  config.json 中 api_key 不能为空")
        sys.exit(1)
    
    print(f"Loaded config from {config_path}: model={model_name}")
    print(f"Using model: {model_name}")
    return base_url, api_key, model_name


def main():
    parser = argparse.ArgumentParser(description='LJP智能体')
    parser.add_argument('--mode', type=str, default='baseline', 
                       choices=['baseline', 'positive-only', 'positive-negative'],
                       help='运行模式')
    parser.add_argument('--input', type=str, help='输入案件文件（json格式）')
    parser.add_argument('--fact', type=str, help='直接输入案件事实文本')
    args = parser.parse_args()
    
    # 加载API配置
    base_url, api_key, model_name = load_api_config()
    
    # 初始化对应模型
    if args.mode == 'baseline':
        agent = ZeroShotLJPBaseline(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name
        )
    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented yet")
    
    # 读取输入案件
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
            case = Case(
                fact=data.get('fact', ''),
                charges=[],
                articles=[],
                judgment='',
                is_positive=False
            )
    elif args.fact:
        case = Case(
            fact=args.fact,
            charges=[],
            articles=[],
            judgment='',
            is_positive=False
        )
    else:
        print("请提供--input或--fact参数")
        return
    
    # 预测
    print(f"\n=== Running {args.mode} mode ===")
    print(f"输入案件事实:\n{case.fact[:500]}...\n" if len(case.fact) > 500 else f"输入案件事实:\n{case.fact}\n")
    
    result = agent.predict(case)
    
    print("=== 预测结果 ===")
    print(f"推理过程:\n{result.predicted_judgment.split('}')[0] if '}' in result.predicted_judgment else result.predicted_judgment}\n")
    print(f"预测罪名: {result.predicted_charges}")
    print(f"预测法条: {result.predicted_articles}")
    print(f"\nToken使用: prompt={result.prompt_tokens}, completion={result.completion_tokens}")


if __name__ == "__main__":
    main()
