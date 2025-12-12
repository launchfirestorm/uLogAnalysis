#!/usr/bin/env python3
"""
ULog Field Lister

Simple script to list all available fields in one or more ULog files.
Saves the output to a text file for easy reference.

Usage:
    python list_ulog_fields.py <file_or_directory> [--output <output_file>]

Examples:
    python list_ulog_fields.py my_log.ulg
    python list_ulog_fields.py ./logs_folder/
    python list_ulog_fields.py my_log.ulg --output fields.txt

Author: Generated for quick ULog inspection
Date: December 11, 2025
"""

import sys
import os
import argparse
from pathlib import Path
from pyulog import ULog


def list_ulog_fields(ulg_file, output_file=None):
    """List all available fields in a ULog file
    
    Args:
        ulg_file: Path to .ulg file
        output_file: Optional file handle to write output to
    """
    try:
        print(f"Loading: {ulg_file.name}")
        ulog = ULog(str(ulg_file))
        
        output = []
        output.append("=" * 80)
        output.append(f"ULog File: {ulg_file.name}")
        output.append("=" * 80)
        output.append(f"Total datasets: {len(ulog.data_list)}")
        output.append("")
        
        for idx, dataset in enumerate(ulog.data_list, 1):
            # Calculate sample rate
            timestamp = dataset.data['timestamp']
            duration = (timestamp[-1] - timestamp[0]) / 1e6
            sample_rate = len(timestamp) / duration if duration > 0 else 0
            
            output.append(f"{idx}. Dataset: '{dataset.name}' (instance {dataset.multi_id})")
            output.append(f"   Samples: {len(timestamp)}, Rate: {sample_rate:.2f} Hz")
            output.append(f"   Fields:")
            
            # List all fields
            for field_name in dataset.data.keys():
                if field_name != 'timestamp':
                    data = dataset.data[field_name]
                    if len(data) > 0:
                        min_val = float(data.min())
                        max_val = float(data.max())
                        output.append(f"      - {field_name}: [{min_val:.4f}, {max_val:.4f}]")
                    else:
                        output.append(f"      - {field_name}: [no data]")
            
            output.append("")
        
        # Print to console
        for line in output:
            print(line)
        
        # Write to file if specified
        if output_file:
            for line in output:
                output_file.write(line + '\n')
            output_file.write('\n')
        
        return True
        
    except Exception as e:
        print(f"Error processing {ulg_file.name}: {e}")
        return False


def process_path(input_path, output_path=None):
    """Process a file or directory of ULog files
    
    Args:
        input_path: Path to .ulg file or directory
        output_path: Optional output file path
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        return
    
    # Collect all .ulg files
    ulg_files = []
    if input_path.is_file():
        if input_path.suffix == '.ulg':
            ulg_files.append(input_path)
        else:
            print(f"Error: File must have .ulg extension: {input_path}")
            return
    else:
        # Directory - find all .ulg files recursively
        ulg_files = sorted(input_path.rglob('*.ulg'))
        if not ulg_files:
            print(f"Error: No .ulg files found in: {input_path}")
            return
    
    print(f"\nFound {len(ulg_files)} ULog file(s)")
    print()
    
    # Open output file if specified
    output_file = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(output_path, 'w')
        print(f"Writing output to: {output_path}")
        print()
    
    # Process each file
    success_count = 0
    for ulg_file in ulg_files:
        if list_ulog_fields(ulg_file, output_file):
            success_count += 1
    
    if output_file:
        output_file.close()
        print(f"\nOutput saved to: {output_path}")
    
    print(f"\nProcessed {success_count}/{len(ulg_files)} files successfully")


def main():
    parser = argparse.ArgumentParser(
        description="List all available fields in ULog file(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python list_ulog_fields.py my_log.ulg
  python list_ulog_fields.py ./logs_folder/
  python list_ulog_fields.py my_log.ulg --output fields.txt
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Path to .ulg file or directory containing .ulg files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (default: print to console only)'
    )
    
    args = parser.parse_args()
    
    process_path(args.input, args.output)


if __name__ == "__main__":
    main()
