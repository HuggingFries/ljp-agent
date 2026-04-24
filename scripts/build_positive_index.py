#!/usr/bin/env python3
"""
Build FAISS index for positive cases using L1 embeddings.
Reads hierarchical positive cases from positive_cases/collected_positive_hierarchical.json,
embeds L1 fields, and saves index files under data/index_hierarchical/ with prefix "pos".

Usage:
  python scripts/build_positive_index.py [--config CONFIG] [--overwrite]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_api_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_l1_text(l1_data: dict) -> str:
    """Concatenate all L1 fields into a single string for embedding."""
    fields = [
        l1_data.get("犯罪主体", ""),
        l1_data.get("犯罪行为", ""),
        l1_data.get("犯罪手段", ""),
        l1_data.get("犯罪客体", ""),
        l1_data.get("犯罪动机", ""),
        l1_data.get("危害程度", ""),
        l1_data.get("法益类型", ""),
    ]
    # filter empty strings
    non_empty = [f for f in fields if f.strip()]
    return " ".join(non_empty) if non_empty else ""


def main():
    parser = argparse.ArgumentParser(description="Build positive index from hierarchical positive cases")
    parser.add_argument("--config", default="config/config.yaml", help="Config YAML file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index files")
    args = parser.parse_args()

    # Load config
    config = load_api_config(args.config)
    index_config = config.get("index", {})
    embedding_model_name = index_config.get("embedding_model", "uer/sbert-base-chinese-nli")
    device = index_config.get("device", "cpu")  # 'cpu' or 'cuda'

    # Paths
    base_dir = Path(__file__).resolve().parent.parent
    input_file = base_dir / "data" / "positive_cases" / "collected_positive_hierarchical.json"
    output_dir = base_dir / "data" / "index_hierarchical"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_cases = output_dir / "pos_hierarchical_cases.json"
    output_emb = output_dir / "pos_l1_embeddings.npy"
    output_meta = output_dir / "pos_metadata.json"

    # Check for existing files
    if output_cases.exists() or output_emb.exists() or output_meta.exists():
        if not args.overwrite:
            logger.error("Index files already exist. Use --overwrite to replace them.")
            sys.exit(1)
        else:
            logger.warning("Overwriting existing index files.")

    # Load positive hierarchical cases
    if not input_file.exists():
        logger.error(f"Positive hierarchical file not found: {input_file}")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    logger.info(f"Loaded {len(cases)} positive hierarchical cases from {input_file}")

    # Prepare data
    l1_texts = []
    metadata = []
    clean_cases = []  # store full case data without embedding

    for idx, case in enumerate(cases):
        l1 = case.get("L1", {})
        if not l1:
            logger.warning(f"Case {idx} has empty L1, skipping")
            continue
        text = build_l1_text(l1)
        if not text:
            logger.warning(f"Case {idx} L1 has no text after concatenation, skipping")
            continue
        l1_texts.append(text)
        metadata.append({
            "index": idx,
            "charge": case.get("L0", {}).get("charge", ""),
            "true_charges": case.get("L0", {}).get("true_charges", []),
        })
        clean_cases.append(case)  # keep original full case

    logger.info(f"Valid cases for indexing: {len(l1_texts)}")

    if not l1_texts:
        logger.error("No valid L1 texts found.")
        sys.exit(1)

    # Load embedding model
    logger.info(f"Loading embedding model: {embedding_model_name} on device {device}")
    model = SentenceTransformer(embedding_model_name, device=device)
    logger.info("Generating embeddings...")
    embeddings = model.encode(l1_texts, show_progress_bar=True, convert_to_numpy=True)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Save files
    np.save(output_emb, embeddings)
    logger.info(f"Saved embeddings to {output_emb}")

    with open(output_meta, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metadata to {output_meta}")

    with open(output_cases, 'w', encoding='utf-8') as f:
        json.dump(clean_cases, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved full cases to {output_cases}")

    # Summary
    logger.info("Positive index building completed.")
    logger.info(f"  Number of items: {len(clean_cases)}")
    logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()