#!/usr/bin/env python3
"""
构建分层错误案例的检索索引文件
对collected_errors_hierarchical.json中的每个案例，仅对L1层（事实要素层）做embedding

Usage:
  python scripts/build_hierarchical_index.py [options]

Options:
  --config CONFIG       Config file path (default: config/config.yaml)
  --input PATH          Input hierarchical error cases json (default: data/negative_error_cases/collected_errors_hierarchical_multi.json)
  --output-dir DIR      Output index directory (default: data/index_hierarchical)
  --embedding-model NAME Embedding model name (overrides config.index.embedding_model)
  --device DEVICE       Device to run embedding (default: cpu)
  --max-samples N       Only process first N samples for test (optional)
"""

import argparse
import json
import logging
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

print("👉 Script started, importing dependencies...")

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("🚀 Starting build_hierarchical_index.py")

# Project paths
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent


def main():
    parser = argparse.ArgumentParser(description='Build embedding index for hierarchical negative KB')
    parser.add_argument("--config", default=str(ROOT_DIR / "config/config.yaml"), help="Config file path (YAML)")
    parser.add_argument("--input", default=str(ROOT_DIR / "data/negative_error_cases/collected_errors_hierarchical_multi.json"), help="Input hierarchical error cases json")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "data/index_hierarchical"), help="Output index directory")
    parser.add_argument("--embedding-model", help="Embedding model name (overrides config)")
    parser.add_argument("--device", default="cpu", help="Device to run embedding")
    parser.add_argument("--max-samples", type=int, default=None, help="Only process first N samples for test")
    args = parser.parse_args()
    
    # Read global config (YAML format)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Get embedding model name
    if args.embedding_model:
        model_name = args.embedding_model
    else:
        model_name = config.get("index", {}).get("embedding_model", "uer/sbert-base-chinese-nli")
    
    logger.info(f"Using embedding model: {model_name}")
    logger.info(f"Rule: Only L1 layer (legal elements) is embedded for retrieval. Full case is archived for prompt injection.")
    
    # Load input hierarchical error cases
    logger.info(f"Checking input file: {args.input}")
    if not os.path.exists(args.input):
        logger.error(f"❌ Input file not found: {args.input}")
        logger.error("Did you run build_hierarchical_error_kb.py first?")
        exit(1)
    
    logger.info(f"✅ Input file exists, size: {os.path.getsize(args.input)} bytes")
    logger.info(f"Loading input json...")
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    logger.info(f"✅ JSON loaded, parsing...")
    if isinstance(input_data, dict) and "cases" in input_data:
        cases = input_data["cases"]
        metadata = input_data.get("metadata", {})
        logger.info(f"✅ Found {len(cases)} cases in input")
    else:
        logger.error("❌ Input data format error: expected dict with 'cases' key")
        exit(1)
    
    # Limit samples for testing if requested
    if args.max_samples is not None and args.max_samples > 0:
        cases = cases[:args.max_samples]
        logger.info(f"Limited to first {args.max_samples} samples for testing")
    
    logger.info(f"Loaded {len(cases)} hierarchical cases from {args.input}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isdir(args.output_dir):
        logger.error(f"❌ Failed to create output directory {args.output_dir}")
        exit(1)
    logger.info(f"✅ Output directory ready: {args.output_dir}")
    
    # Load embedding model
    logger.info(f"Loading embedding model {model_name} to {args.device}")
    model = SentenceTransformer(model_name, device=args.device)
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"✅ Model loaded, embedding dimension: {dim}")
    
    # Process negative cases: ONLY L1.legal_elements is embedded for retrieval
    l1_texts: List[str] = []
    output_cases: List[Dict[str, Any]] = []
    
    logger.info(f"Processing {len(cases)} cases, extracting L1 legal elements...")
    for idx, item in enumerate(cases):
        # item has L0/L1/L2/L3 structure from build_hierarchical_error_kb.py
        L1 = item.get("L1", {})
        
        # Get L1 legal elements text for embedding (NEW structure: L1.legal_elements is the 7要素 dict)
        legal_elements = L1.get("legal_elements", {})
        
        parts = []
        for name, value in legal_elements.items():
            if value and isinstance(value, str) and value.strip():
                parts.append(f"{name}：{value.strip()}")
        
        l1_text = "\n".join(parts)
        l1_texts.append(l1_text)
        output_cases.append(item)  # Keep full structure (all layers) for output, used in prompt injection
    
    logger.info(f"✅ Extracted L1 text for {len(output_cases)} negative cases")
    logger.info(f"Generating embeddings for L1 layer only... (this may take a few minutes)")
    
    # Compute embeddings (normalize for cosine similarity retrieval)
    l1_embeddings = model.encode(l1_texts, normalize_embeddings=True, show_progress_bar=True)
    
    # Save outputs
    prefix = "neg"
    cases_path = os.path.join(args.output_dir, f"{prefix}_hierarchical_cases.json")
    embeddings_path = os.path.join(args.output_dir, f"{prefix}_l1_embeddings.npy")
    
    # Save full cases (keep all layers L0-L3 for query time prompt injection)
    with open(cases_path, 'w', encoding='utf-8') as f:
        json.dump(output_cases, f, ensure_ascii=False, indent=2)
    
    # Save only L1 embeddings for fast retrieval
    np.save(embeddings_path, l1_embeddings)
    
    # Save metadata with config parameters
    meta_output = metadata.copy()
    meta_output.update({
        "embedding_model": model_name,
        "num_cases": len(output_cases),
        "embedding_dim": dim,
        "retrieval_rule": "Only L1 legal elements layer is embedded for retrieval. Target query combines extracted elements + original fact with weight",
        "elements_weight": config["retriever"]["negative"].get("elements_weight", 0.7),
        "fact_weight": config["retriever"]["negative"].get("fact_weight", 0.3),
    })
    meta_path = os.path.join(args.output_dir, f"{prefix}_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved negative index to {args.output_dir}:")
    logger.info(f"  Full cases (all layers): {cases_path}")
    logger.info(f"  L1 embeddings (for retrieval): {embeddings_path} shape: {l1_embeddings.shape}")
    logger.info(f"  Metadata: {meta_path}")
    
    logger.info("\n✅ Done! Negative index built successfully.")


if __name__ == "__main__":
    import yaml
    main()
