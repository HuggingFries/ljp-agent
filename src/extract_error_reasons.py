#!/usr/bin/env python3
"""
Remove error_reason field from collected errors file, keep all other fields unchanged.
"""

import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/negative_error_cases/collected_errors_with_reasons.json',
                        help='Input file path')
    parser.add_argument('--output', type=str, default='data/negative_error_cases/collected_errors.json',
                        help='Output file path')
    args = parser.parse_args()
    
    # Resolve relative paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.input):
        args.input = os.path.join(project_root, args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.join(project_root, args.output)
    
    # Load input data
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process error_cases: remove error_reason field from each item
    if 'error_cases' in data:
        data['error_cases'] = [
            {k: v for k, v in item.items() if k != 'error_reason'}
            for item in data['error_cases']
        ]
    elif 'errors' in data:
        data['errors'] = [
            {k: v for k, v in item.items() if k != 'error_reason'}
            for item in data['errors']
        ]
    else:
        # If toplevel is list of items
        data = [
            {k: v for k, v in item.items() if k != 'error_reason'}
            for item in data
        ]
    
    # Save output, keep all other metadata unchanged
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Count items for info
    if 'error_cases' in data:
        count = len(data['error_cases'])
    elif 'errors' in data:
        count = len(data['errors'])
    else:
        count = len(data)
    
    print(f"Removed error_reason field from {count} items, saved to {args.output}")

if __name__ == '__main__':
    main()
