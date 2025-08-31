import pandas as pd
import glob
import os

# Path where your CSV files are stored
path = "C:/Users/Michelle/Downloads/Chosen"  

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(path, "*.csv"))
csv_files.sort()  # optional: sort by filename for consistent order

dfs = []

for i, file in enumerate(csv_files):
    if i == 0:
        # First file: keep the header
        df = pd.read_csv(file)
    else:
        # Other files: skip the header row
        df = pd.read_csv(file, header = 0)

    dfs.append(df)

# Concatenate into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Save final CSV with header only once
combined_df.to_csv("C:/Users/Michelle/Downloads/combined.csv", index=False)

print(f"✅ Combined {len(csv_files)} CSV files into one file: combined.csv")
