"""
评估Zero-Shot Baseline在CAIL2018测试集上的准确率
计算指标：
- 罪名准确率（Acc@charge）
- 法条准确率（Acc@article）
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Tuple
from tqdm import tqdm

from agent import Case, DataLoader
from baseline import ZeroShotLJPBaseline
from run_agent import load_api_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(
    model: ZeroShotLJPBaseline,
    test_data: List[dict],
    test_file: str,
    max_samples: int = None
) -> Tuple[float, float, List[dict]]:
    """评估模型，同时保存每个样本的期望输出和实际输出"""
    correct_charges = 0
    correct_articles = 0
    total = 0
    details = []  # 保存每个样本的详细对比
    
    # 随机采样，避免相同罪名扎堆在前N个
    if max_samples is not None and max_samples < len(test_data):
        import random
        random.seed(42)  # 固定种子保证可复现
        test_data = random.sample(test_data, max_samples)
    
    for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
        case = DataLoader.convert_to_case(item, is_positive=False)
        
        # 获取真实标签
        # CAIL2018中charge/article是编号索引，对应accu.txt/law.txt
        # 我们先尝试读取标签映射
        if not hasattr(model, 'charge_names'):
            # 惰性加载标签映射 - 正确路径在baseline子目录
            charge_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/accu.txt')
            if not os.path.exists(charge_path):
                # 如果找不到，尝试相对路径
                charge_path = os.path.join(os.path.dirname(test_file), '../baseline/accu.txt')
            if os.path.exists(charge_path):
                with open(charge_path, 'r', encoding='utf-8') as f:
                    # accu.txt每行就是一个罪名，顺序就是索引(0开始)
                    model.charge_names = [line.strip() for line in f if line.strip()]
            else:
                logger.warning(f"accu.txt not found at {charge_path}, will use raw ids")
                model.charge_names = None
        
        if not hasattr(model, 'article_names'):
            article_path = os.path.join(os.path.dirname(__file__), 'data/cail2018/baseline/law.txt')
            if not os.path.exists(article_path):
                article_path = os.path.join(os.path.dirname(test_file), '../baseline/law.txt')
            if os.path.exists(article_path):
                with open(article_path, 'r', encoding='utf-8') as f:
                    # law.txt每行就是法条编号，顺序就是索引(0开始)，法条编号就是内容
                    model.article_names = [line.strip() for line in f if line.strip()]
            else:
                logger.warning(f"law.txt not found at {article_path}, will use raw ids")
                model.article_names = None
        
        # CAIL2018支持两种数据格式：标签在顶层或meta内
        true_charges = set()
        true_charges_list = []
        true_articles = set()
        true_articles_list = []
        
        if 'meta' in item and (item['meta'].get('accusation') or item['meta'].get('relevant_articles')):
            # 比赛json格式：标签在meta里，已经是罪名名称和法条编号（当前测试数据是这种）
            true_charge_names = item['meta'].get('accusation', [])
            true_article_ids = item['meta'].get('relevant_articles', [])
            
            for name in true_charge_names:
                true_charges.add(name.strip())
                true_charges_list.append(name.strip())
            
            for aid in true_article_ids:
                aid = str(aid).strip()
                true_articles.add(aid)
                true_articles_list.append({"id": aid, "name": ""})
        
        else:
            # 原始索引格式：charge/article是索引编号，需要从accu.txt/law.txt映射
            true_charge_indices = item.get('charge', [])
            true_article_indices = item.get('article', [])
            
            for idx in true_charge_indices:
                try:
                    if hasattr(model, 'charge_names') and model.charge_names and idx >= 0 and idx < len(model.charge_names):
                        name = model.charge_names[idx]
                        true_charges.add(name)
                        true_charges_list.append(name)
                    else:
                        true_charges.add(str(idx))
                        true_charges_list.append(str(idx))
                except:
                    true_charges.add(str(idx))
                    true_charges_list.append(str(idx))
            
            for idx in true_article_indices:
                try:
                    if hasattr(model, 'article_names') and model.article_names and idx >= 0 and idx < len(model.article_names):
                        article_id = model.article_names[idx]
                        true_articles.add(article_id)
                        true_articles_list.append({"id": article_id, "name": ""})
                    else:
                        article_id = str(idx)
                        true_articles.add(article_id)
                        true_articles_list.append({"id": article_id, "name": ""})
                except:
                    article_id = str(idx)
                    true_articles.add(article_id)
                    true_articles_list.append({"id": article_id, "name": ""})
        
        detail = {
            "index": i,
            "fact": case.fact,
            "true_charges": true_charges_list,
            "true_articles": true_articles_list,
            "predicted_charges": [],
            "predicted_articles": [],
            "predicted_judgment": "",
            "correct_charge": False,
            "correct_article": False,
            "error": None
        }
        
        def clean_charge(charge: str) -> str:
            """清洗罪名：去掉末尾的"罪"字"""
            charge = charge.strip()
            if charge.endswith("罪"):
                charge = charge[:-1]
            return charge
        
        def clean_article(article: str) -> str:
            """清洗法条：提取编号，去掉多余文字"""
            article = article.strip()
            # 去掉所有非数字字符，只保留数字
            digits = ''.join([c for c in article if c.isdigit()])
            if digits:
                return digits
            # 如果没提取到数字，返回原字符串去掉前后缀
            article = article.replace("《中华人民共和国刑法》", "").replace("第", "").replace("条", "").replace("款", "").replace("第一款", "").strip()
            digits = ''.join([c for c in article if c.isdigit()])
            return digits if digits else article
        
        try:
            result = model.predict(case)
            
            # 预测集合：清洗格式，和真实标签对齐
            pred_charges = set(map(clean_charge, result.predicted_charges))
            pred_articles = set(map(clean_article, result.predicted_articles))
            
            # 保存预测结果
            detail["predicted_charges"] = list(pred_charges)
            detail["predicted_articles"] = list(pred_articles)
            detail["predicted_judgment"] = result.predicted_judgment
            
            # 判断是否完全正确（所有罪名/法条都预测对才算对）
            charge_correct = pred_charges == true_charges
            article_correct = pred_articles == true_articles
            
            if charge_correct:
                correct_charges += 1
            if article_correct:
                correct_articles += 1
            
            detail["correct_charge"] = charge_correct
            detail["correct_article"] = article_correct
            
            total += 1
            
            # 每10个例子打印一次进度
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(test_data)}, "
                           f"Charge Acc: {correct_charges/total:.4f}, "
                           f"Article Acc: {correct_articles/total:.4f}")
        
        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            detail["error"] = str(e)
        
        details.append(detail)
    
    acc_charge = correct_charges / total if total > 0 else 0
    acc_article = correct_articles / total if total > 0 else 0
    
    return acc_charge, acc_article, details


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline on CAIL2018')
    parser.add_argument('--test-file', type=str, default='data/final_all_data/first_stage/test.json',
                       help='CAIL2018 test.json path')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum number of samples to evaluate (for quick test)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    args = parser.parse_args()
    
    # 加载API配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    base_url, api_key, model_name = load_api_config(config)
    
    # 初始化模型
    model = ZeroShotLJPBaseline(base_url=base_url, api_key=api_key, model_name=model_name)
    
    # 加载测试数据
    test_data = DataLoader.load_cail2018(args.test_file)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # 开始评估
    logger.info(f"Starting evaluation, max_samples={args.max_samples}")
    acc_charge, acc_article, details = evaluate(model, test_data, args.test_file, args.max_samples)
    
    # 输出结果
    print("\n" + "="*50)
    print("Evaluation Result (Zero-Shot Baseline):")
    print(f"Total evaluated: {min(args.max_samples, len(test_data))} samples")
    print(f"Charge Accuracy: {acc_charge:.4f} ({acc_charge*100:.2f}%)")
    print(f"Article Accuracy: {acc_article:.4f} ({acc_article*100:.2f}%)")
    print("="*50)
    
    # 保存结果（包含详细对比）
    os.makedirs("results", exist_ok=True)
    output_file = os.path.join("results", f"result_baseline_{args.max_samples}.json")
    details_file = os.path.join("results", f"details_baseline_{args.max_samples}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "mode": "zero-shot-baseline",
            "max_samples": args.max_samples,
            "acc_charge": acc_charge,
            "acc_article": acc_article
        }, f, indent=2, ensure_ascii=False)
    
    # 保存详细对比（每个样本的期望输出和实际输出）
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump({
            "details": details
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Aggregate result saved to {output_file}")
    logger.info(f"Detailed predictions saved to {details_file}")


if __name__ == "__main__":
    main()
