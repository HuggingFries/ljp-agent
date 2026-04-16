#!/usr/bin/env python3
"""
Data cleaning script: Remove invalid error cases
Cleaning rules:
1. Model output infinite loop cut off: detected by short/empty error_reason or too long predicted_judgment
2. Invalid fact without enough criminal information: detected by empty predicted_judgment or empty predicted_charges
3. Move invalid cases to rest_data field at end of file, keep original structure
"""

import json
import argparse
from typing import List, Dict, Any


def clean_error_cases(input_path: str, output_path: str) -> None:
    """
    Clean error case data
    
    Args:
        input_path: Input JSON file path
        output_path: Output cleaned JSON file path
    """
    # Load original data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get error_cases list
    error_cases: List[Dict[str, Any]] = data.get('error_cases', [])
    if not error_cases:
        print("error_cases field not found, exiting")
        return
    
    cleaned_cases: List[Dict[str, Any]] = []
    rest_data: List[Dict[str, Any]] = []
    
    # Statistics for cleaning rules
    stats = {
        'total': len(error_cases),
        'empty_error_reason': 0,
        'empty_predicted_judgment': 0,
        'short_error_reason': 0,
        'infinite_loop_prediction': 0,
        'empty_predicted_charges': 0,
        'kept': 0,
        'removed': 0
    }
    
    for case in error_cases:
        remove = False
        reason = ""
        
        # Rule 1: error_reason is None or empty
        if 'error_reason' not in case or case['error_reason'] is None or case['error_reason'].strip() == "":
            stats['empty_error_reason'] += 1
            remove = True
            reason = "empty_error_reason"
        # Rule 1 extension: error_reason too short (may be cut off), threshold 50 chars
        elif len(case['error_reason'].strip()) < 50:
            stats['short_error_reason'] += 1
            remove = True
            reason = "short_error_reason (length < 50)"
        
        # Rule 2: predicted_judgment is None or empty
        if not remove:
            if 'predicted_judgment' not in case or case['predicted_judgment'] is None or case['predicted_judgment'].strip() == "":
                stats['empty_predicted_judgment'] += 1
                remove = True
                reason = "empty_predicted_judgment"
            # Rule 2 extension: predicted_judgment too long (> 3000 chars) → infinite loop cut off
            elif len(case['predicted_judgment']) > 3000:
                stats['infinite_loop_prediction'] += 1
                remove = True
                reason = "infinite_loop_prediction (length > 3000)"
        
        # Rule 3: predicted_charges is empty array → model output nothing, remove
        if not remove:
            if 'predicted_charges' not in case or not isinstance(case['predicted_charges'], list) or len(case['predicted_charges']) == 0:
                stats['empty_predicted_charges'] += 1
                remove = True
                reason = "empty_predicted_charges (empty array)"
        
        # Sort and save
        if remove:
            case['remove_reason'] = reason
            rest_data.append(case)
            stats['removed'] += 1
        else:
            cleaned_cases.append(case)
            stats['kept'] += 1
    
    # Update data structure
    data['error_cases'] = cleaned_cases
    data['rest_data'] = rest_data
    
    # Update top-level count
    data['count'] = len(cleaned_cases)
    if 'metadata' in data:
        data['metadata']['count_after_clean'] = len(cleaned_cases)
        data['metadata']['removed_count'] = len(rest_data)
    
    # Save result
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print("=" * 50)
    print("Data cleaning completed")
    print(f"Total original cases: {stats['total']}")
    print(f"Kept valid cases: {stats['kept']}")
    print(f"Removed invalid cases: {stats['removed']}")
    print("-" * 30)
    print(f" - empty error_reason: {stats['empty_error_reason']}")
    print(f" - short error_reason (<50 chars): {stats['short_error_reason']}")
    print(f" - empty predicted_judgment: {stats['empty_predicted_judgment']}")
    print(f" - infinite loop prediction (>3000 chars): {stats['infinite_loop_prediction']}")
    print(f" - empty predicted_charges array: {stats['empty_predicted_charges']}")
    print("=" * 50)
    print(f"Result saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Clean invalid error case data')
    parser.add_argument(
        '-i', '--input', 
        default='data/negative_error_cases/collected_errors_with_reasons.json',
        help='Input JSON file path (default: collected_errors_with_reasons.json)'
    )
    parser.add_argument(
        '-o', '--output', 
        default='data/negative_error_cases/collected_errors_with_reasons.json',
        help='Output JSON file path (default: collected_errors_with_reasons.json, overwrite input)'
    )
    args = parser.parse_args()
    
    clean_error_cases(args.input, args.output)


if __name__ == '__main__':
    main()
