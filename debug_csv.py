#!/usr/bin/env python3
"""
Debug script to check CSV formatting issues
"""

import pandas as pd
from pathlib import Path

# Check what our API creates
test_data = {
    "business_name": "Joe's Pizza",
    "author_name": "John",
    "rating": 4.0,
    "text": "Good pizza.",
    "description": "Pizza restaurant.",
    "pics": "no",
    "time": 1756544572948
}

print("Original data:")
print(test_data)

# Create DataFrame like the API does
df = pd.DataFrame([test_data])
print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Save with semicolon separator like API does
temp_file = "debug_input.csv"
df.to_csv(temp_file, index=False, encoding='utf-8', sep=';')

print(f"\nSaved CSV content:")
with open(temp_file, 'r', encoding='utf-8') as f:
    content = f.read()
    print(repr(content))

# Try to read it back
print(f"\nReading back:")
df_read = pd.read_csv(temp_file, sep=';', encoding='utf-8')
print(f"Shape after reading: {df_read.shape}")
print(f"Columns after reading: {list(df_read.columns)}")
print(f"First row: {df_read.iloc[0].to_dict()}")

# Clean up
Path(temp_file).unlink()
