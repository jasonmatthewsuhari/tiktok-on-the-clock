#!/usr/bin/env python3
"""
Debug script to trace Stage 1 processing
"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Create test input like the API
test_data = {
    "business_name": "Joe's Pizza",
    "author_name": "John",
    "rating": 4.0,
    "text": "Good pizza.",
    "description": "Pizza restaurant.",
    "pics": "no",
    "time": 1756544572948
}

df = pd.DataFrame([test_data])
temp_dir = Path("temp_debug")
temp_dir.mkdir(exist_ok=True)

input_file = temp_dir / "input.csv"
output_file = temp_dir / "output.csv"

# Save input like API does
df.to_csv(input_file, index=False, encoding='utf-8', sep=';')
print(f"Created input file: {input_file}")

print(f"\nInput file content:")
with open(input_file, 'r', encoding='utf-8') as f:
    print(repr(f.read()))

# Test reading the input
print(f"\nReading input file:")
df_read = pd.read_csv(input_file, sep=';', encoding='utf-8')
print(f"Shape: {df_read.shape}")
print(f"Columns: {list(df_read.columns)}")

# Import Stage 1 and run it
try:
    import importlib
    module = importlib.import_module('src.01_take_input_csv')
    run = module.run
    
    config = {
        'input_file': str(input_file),
        'output_file': str(output_file),
        'execution_id': 'debug'
    }
    
    print(f"\nRunning Stage 1...")
    success = run(config)
    print(f"Stage 1 success: {success}")
    
    if output_file.exists():
        print(f"\nOutput file content:")
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(repr(content))
        
        print(f"\nReading output file:")
        df_out = pd.read_csv(output_file, sep=';', encoding='utf-8')
        print(f"Output shape: {df_out.shape}")
        print(f"Output columns: {list(df_out.columns)}")
    else:
        print("No output file created")
        
except Exception as e:
    print(f"Error running Stage 1: {e}")
    import traceback
    traceback.print_exc()

# Clean up
try:
    if input_file.exists():
        input_file.unlink()
    if output_file.exists():
        output_file.unlink()
    temp_dir.rmdir()
except:
    pass
