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
    支持DeepSeek/Volcengine等兼容OpenAI格式的API
    优先级：环境变量 > 配置文件
    
    DeepSeek默认配置：
    - DEEPSEEK_BASE_URL: https://api.deepseek.com/v1
    - DEEPSEEK_MODEL_ID: deepseek-chat / deepseek-coder
    """
    # 支持两种环境变量前缀，DeepSeek优先
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model_name = os.environ.get("DEEPSEEK_MODEL_ID", "deepseek-chat")
    
    # 如果没配置DeepSeek，回退到Volcengine
    if api_key is None:
        api_key = os.environ.get("VOLCENGINE_API_KEY")
        base_url = os.environ.get("VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
        model_name = os.environ.get("VOLCENGINE_MODEL_ID", "ark-code-latest")
    
    if api_key is None:
        # 尝试从配置文件读，优先读.deepseek_api，再读.volcengine_api
        api_key_path = os.environ.get("DEEPSEEK_API_KEY_FILE", "/home/node/projects/.deepseek_api")
        if not os.path.exists(api_key_path):
            api_key_path = os.environ.get("VOLCENGINE_API_KEY_FILE", "/home/node/projects/.volcengine_api")
        if os.path.exists(api_key_path):
            with open(api_key_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                api_key = lines[0]
                if len(lines) >= 2:
                    model_name = lines[1]
        else:
            print("⚠️  找不到API Key，请设置环境变量：")
            print("   DeepSeek:")
            print("   export DEEPSEEK_API_KEY=your-api-key")
            print("   export DEEPSEEK_MODEL_ID=deepseek-chat")
            print("或者把api key和model id写入 /home/node/projects/.deepseek_api")
            sys.exit(1)
    
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
