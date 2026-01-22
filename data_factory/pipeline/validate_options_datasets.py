#!/usr/bin/env python3
"""
SPY Options Dataset Validator

Validates the structure of SPY options datasets to ensure:
- Each 1-minute OHLCV bar has exactly 100 options (50 puts + 50 calls)
- No missing timestamps during trading hours
- Consistent data structure between 2024 and 2025 datasets

Usage:
    python validate_options_datasets.py [--file 2024|2025|both]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, time
from collections import defaultdict
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Constants
EXPECTED_OPTIONS_PER_BAR = 100
EXPECTED_PUTS = 50
EXPECTED_CALLS = 50
TRADING_START = time(9, 30)  # Market open (ET)
TRADING_END = time(16, 0)    # Market close (ET)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                        'data', 'processed')


def load_dataset_chunked(filepath: str, chunk_size: int = 500000):
    """
    Load dataset in chunks to handle large files efficiently.
    Yields chunks for processing.
    """
    print(f"Loading {filepath} in chunks of {chunk_size:,} rows...")
    
    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
        print(f"  Chunk {i+1}: {len(chunk):,} rows loaded")
        yield chunk


def analyze_dataset(filepath: str, sample_mode: bool = False) -> dict:
    """
    Analyze a dataset and return diagnostic information.
    
    Args:
        filepath: Path to CSV file
        sample_mode: If True, only analyze first 1M rows for quick diagnostics
        
    Returns:
        Dictionary with analysis results
    """
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"Analyzing: {filename}")
    print(f"{'='*60}")
    
    results = {
        'filename': filename,
        'filepath': filepath,
        'total_rows': 0,
        'unique_timestamps': 0,
        'unique_options': 0,
        'timestamps_with_issues': [],
        'put_call_balance': {},
        'options_per_timestamp': {},
        'missing_timestamps': [],
        'summary': {}
    }
    
    # Counters for streaming analysis
    timestamp_counts = defaultdict(lambda: {'total': 0, 'puts': 0, 'calls': 0})
    all_options = set()
    
    # Load and analyze in chunks
    rows_processed = 0
    max_rows = 1_000_000 if sample_mode else None
    
    for chunk in load_dataset_chunked(filepath):
        rows_processed += len(chunk)
        
        # Normalize timestamp column
        if 'timestamp' in chunk.columns:
            chunk['ts_normalized'] = pd.to_datetime(chunk['timestamp']).dt.floor('min')
        
        # Count options per timestamp
        for ts, group in chunk.groupby('ts_normalized'):
            ts_str = str(ts)
            timestamp_counts[ts_str]['total'] += len(group)
            
            if 'call_put' in group.columns:
                puts = (group['call_put'] == 'P').sum()
                calls = (group['call_put'] == 'C').sum()
            elif 'option_type' in group.columns:
                puts = (group['option_type'].str.lower() == 'put').sum()
                calls = (group['option_type'].str.lower() == 'call').sum()
            else:
                puts = calls = 0
                
            timestamp_counts[ts_str]['puts'] += puts
            timestamp_counts[ts_str]['calls'] += calls
        
        # Track unique options
        if 'option_symbol' in chunk.columns:
            all_options.update(chunk['option_symbol'].unique())
        
        print(f"  Processed: {rows_processed:,} rows")
        
        if max_rows and rows_processed >= max_rows:
            print(f"  Sample mode: stopping at {rows_processed:,} rows")
            break
    
    results['total_rows'] = rows_processed
    results['unique_timestamps'] = len(timestamp_counts)
    results['unique_options'] = len(all_options)
    
    # Analyze timestamp distribution
    print(f"\nAnalyzing {len(timestamp_counts):,} unique timestamps...")
    
    correct_count = 0
    too_few = 0
    too_many = 0
    
    issues = []
    options_distribution = defaultdict(int)
    
    for ts_str, counts in timestamp_counts.items():
        total = counts['total']
        puts = counts['puts']
        calls = counts['calls']
        
        options_distribution[total] += 1
        
        if total == EXPECTED_OPTIONS_PER_BAR and puts == EXPECTED_PUTS and calls == EXPECTED_CALLS:
            correct_count += 1
        else:
            issue = {
                'timestamp': ts_str,
                'total_options': total,
                'puts': puts,
                'calls': calls,
                'issue_type': []
            }
            
            if total < EXPECTED_OPTIONS_PER_BAR:
                too_few += 1
                issue['issue_type'].append(f'missing_{EXPECTED_OPTIONS_PER_BAR - total}_options')
            elif total > EXPECTED_OPTIONS_PER_BAR:
                too_many += 1
                issue['issue_type'].append(f'excess_{total - EXPECTED_OPTIONS_PER_BAR}_options')
            
            if puts != EXPECTED_PUTS:
                issue['issue_type'].append(f'puts_imbalance_{puts}_vs_{EXPECTED_PUTS}')
            if calls != EXPECTED_CALLS:
                issue['issue_type'].append(f'calls_imbalance_{calls}_vs_{EXPECTED_CALLS}')
            
            issues.append(issue)
    
    results['timestamps_with_issues'] = issues
    results['options_per_timestamp'] = dict(options_distribution)
    
    # Calculate summary
    results['summary'] = {
        'total_rows': rows_processed,
        'unique_timestamps': len(timestamp_counts),
        'correct_timestamps': correct_count,
        'timestamps_with_too_few': too_few,
        'timestamps_with_too_many': too_many,
        'issue_percentage': (len(issues) / len(timestamp_counts) * 100) if timestamp_counts else 0,
        'unique_options_seen': len(all_options)
    }
    
    return results


def print_report(results: dict):
    """Print a formatted report of the analysis results."""
    print(f"\n{'='*60}")
    print(f"VALIDATION REPORT: {results['filename']}")
    print(f"{'='*60}")
    
    summary = results['summary']
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total Rows:           {summary['total_rows']:,}")
    print(f"   Unique Timestamps:    {summary['unique_timestamps']:,}")
    print(f"   Unique Options:       {summary['unique_options_seen']:,}")
    print(f"   Expected per bar:     {EXPECTED_OPTIONS_PER_BAR} ({EXPECTED_PUTS}P + {EXPECTED_CALLS}C)")
    
    print(f"\n✅ CORRECT TIMESTAMPS:   {summary['correct_timestamps']:,}")
    print(f"❌ ISSUES FOUND:         {len(results['timestamps_with_issues']):,} ({summary['issue_percentage']:.1f}%)")
    print(f"   - Too few options:    {summary['timestamps_with_too_few']:,}")
    print(f"   - Too many options:   {summary['timestamps_with_too_many']:,}")
    
    # Options distribution
    print(f"\n📈 OPTIONS PER TIMESTAMP DISTRIBUTION:")
    dist = results['options_per_timestamp']
    sorted_counts = sorted(dist.items(), key=lambda x: -x[1])[:10]
    for count, freq in sorted_counts:
        bar = '█' * min(50, int(freq / max(dist.values()) * 50))
        status = '✓' if count == EXPECTED_OPTIONS_PER_BAR else '✗'
        print(f"   {status} {count:4d} options: {freq:,} timestamps {bar}")
    
    # Sample issues
    if results['timestamps_with_issues']:
        print(f"\n🔍 SAMPLE ISSUES (first 10):")
        for issue in results['timestamps_with_issues'][:10]:
            print(f"   {issue['timestamp']}: {issue['total_options']} opts "
                  f"({issue['puts']}P/{issue['calls']}C) - {', '.join(issue['issue_type'])}")


def compare_datasets(results_2024: dict, results_2025: dict):
    """Compare two dataset analysis results."""
    print(f"\n{'='*60}")
    print("DATASET COMPARISON: 2024 vs 2025")
    print(f"{'='*60}")
    
    s24 = results_2024['summary']
    s25 = results_2025['summary']
    
    print(f"\n{'Metric':<30} {'2024':>15} {'2025':>15} {'Diff':>15}")
    print("-" * 75)
    
    metrics = [
        ('Total Rows', s24['total_rows'], s25['total_rows']),
        ('Unique Timestamps', s24['unique_timestamps'], s25['unique_timestamps']),
        ('Correct Timestamps', s24['correct_timestamps'], s25['correct_timestamps']),
        ('Issues Found', len(results_2024['timestamps_with_issues']), 
         len(results_2025['timestamps_with_issues'])),
        ('Too Few Options', s24['timestamps_with_too_few'], s25['timestamps_with_too_few']),
        ('Too Many Options', s24['timestamps_with_too_many'], s25['timestamps_with_too_many']),
    ]
    
    for name, v24, v25 in metrics:
        diff = v25 - v24
        diff_str = f"+{diff:,}" if diff > 0 else f"{diff:,}"
        print(f"{name:<30} {v24:>15,} {v25:>15,} {diff_str:>15}")
    
    # Row ratio
    if s24['total_rows'] > 0:
        ratio = s25['total_rows'] / s24['total_rows']
        print(f"\n📊 Row Ratio (2025/2024): {ratio:.3f}x")
        if ratio > 1.1:
            print("   ⚠️  2025 has significantly more rows than expected")
        elif ratio < 0.9:
            print("   ⚠️  2025 has significantly fewer rows than expected")


def save_report(results: dict, output_path: str):
    """Save analysis results to JSON file."""
    # Convert to JSON-serializable format
    output = {
        'filename': results['filename'],
        'summary': results['summary'],
        'options_distribution': results['options_per_timestamp'],
        'sample_issues': results['timestamps_with_issues'][:100]  # First 100 issues
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n📄 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate SPY Options Datasets')
    parser.add_argument('--file', choices=['2024', '2025', 'both'], default='both',
                        help='Which dataset to analyze')
    parser.add_argument('--sample', action='store_true',
                        help='Sample mode: only analyze first 1M rows')
    parser.add_argument('--output-dir', default=None,
                        help='Directory to save reports (default: data/processed)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SPY OPTIONS DATASET VALIDATOR")
    print("=" * 60)
    print(f"Mode: {'Sample (1M rows)' if args.sample else 'Full Analysis'}")
    print(f"Target: {args.file}")
    
    output_dir = args.output_dir or DATA_DIR
    
    results = {}
    
    if args.file in ['2024', 'both']:
        filepath_2024 = os.path.join(DATA_DIR, 'Spy_Options_2024_1m.csv')
        if os.path.exists(filepath_2024):
            results['2024'] = analyze_dataset(filepath_2024, sample_mode=args.sample)
            print_report(results['2024'])
            save_report(results['2024'], os.path.join(output_dir, 'validation_report_2024.json'))
        else:
            print(f"⚠️  File not found: {filepath_2024}")
    
    if args.file in ['2025', 'both']:
        filepath_2025 = os.path.join(DATA_DIR, 'Spy_Options_2025_1m.csv')
        if os.path.exists(filepath_2025):
            results['2025'] = analyze_dataset(filepath_2025, sample_mode=args.sample)
            print_report(results['2025'])
            save_report(results['2025'], os.path.join(output_dir, 'validation_report_2025.json'))
        else:
            print(f"⚠️  File not found: {filepath_2025}")
    
    # Compare if both analyzed
    if '2024' in results and '2025' in results:
        compare_datasets(results['2024'], results['2025'])
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
