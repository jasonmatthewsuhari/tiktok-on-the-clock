#!/usr/bin/env python3
"""
Quick runner script for reviewer distribution analysis
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the reviewer distribution analysis on the input data."""
    
    print("🚀 Running Reviewer Distribution Analysis")
    print("=" * 50)
    
    # Check if input file exists
    input_file = "data/input.csv"
    if not Path(input_file).exists():
        print(f"❌ Error: {input_file} not found!")
        print("Please make sure your input CSV file is in the data/ directory.")
        return
    
    # Run the analysis
    try:
        # For the original column names (name_x instead of author_name)
        cmd = [
            sys.executable, 
            "analyze_reviewer_distribution.py", 
            input_file,
            "--author-col", "name_x",
            "--user-col", "user_id",
            "--output-dir", "reviewer_plots"
        ]
        
        print(f"📊 Analyzing {input_file} with columns:")
        print(f"   Author column: name_x")
        print(f"   User ID column: user_id")
        print(f"   Output directory: reviewer_plots/")
        print()
        
        result = subprocess.run(cmd, check=True)
        
        print("\n🎉 Analysis completed successfully!")
        print("📁 Check the 'reviewer_plots/' directory for:")
        print("   📊 reviewer_distribution_by_author.png")
        print("   📊 reviewer_distribution_by_user_id.png")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analysis: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
