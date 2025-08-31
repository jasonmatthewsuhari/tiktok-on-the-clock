import json
import random
import pandas as pd
import gzip
import os

def sample_json_gz_to_df(file_path, sample_size=1000):
    sampled_data = []
    valid_count = 0  # count only valid rows

    with gzip.open(file_path, "rt", encoding="utf-8") as f:  # "rt" = read text mode
        for line in f:
            obj = json.loads(line)

            if "text" not in obj or obj["text"] is None or str(obj["text"]).strip() == "":
                continue

            if valid_count < sample_size:
                sampled_data.append(obj)
            else:
                r = random.randint(0, valid_count)
                if r < sample_size:
                    sampled_data[r] = obj

            valid_count += 1

    return pd.DataFrame(sampled_data)

def left_merge_with_large_gz(df_left, large_gz_path, key="gmap_id"):
    lookup_keys = set(df_left[key].dropna().astype(str))
    temp_csv = f"temp_matches_{os.path.basename(large_gz_path)}.csv"

    with open(temp_csv, "w", encoding="utf-8") as out:
        # Write header first
        out.write(",".join(df_left.columns) + "\n")

        with gzip.open(large_gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if str(obj.get(key)) in lookup_keys:
                    pd.DataFrame([obj]).to_csv(out, header=False, index=False)

    df_right = pd.read_csv(temp_csv)
    return df_left.merge(df_right, on=key, how="left")

def convert_filename(filename: str, to: str = "meta"):
    """
    Convert between review and meta filenames (handles .json and .json.gz).
    """
    if filename.endswith(".json.gz"):
        base, ext = filename[:-8], ".json.gz"
    elif filename.endswith(".json"):
        base, ext = filename[:-5], ".json"
    else:
        raise ValueError("Filename must end with .json or .json.gz")

    parts = base.split("-")
    if len(parts) < 2:
        raise ValueError("Unexpected filename format")

    prefix = parts[0]
    rest = "-".join(parts[1:])  # everything after first dash

    if to == "meta":
        # review-North_Carolina_10 -> meta-North_Carolina
        if prefix != "review":
            raise ValueError("Expected a review file")
        # Remove trailing number if present
        if "_" in rest and rest.split("_")[-1].isdigit():
            rest = "_".join(rest.split("_")[:-1])
        return f"meta-{rest}{ext}"

    elif to == "review":
        # meta-North_Carolina -> review-North_Carolina_10
        if prefix != "meta":
            raise ValueError("Expected a meta file")
        return f"review-{rest}_10{ext}"

    else:
        raise ValueError("`to` must be 'meta' or 'review'")

review_files = [f for f in os.listdir(".") if f.endswith(".gz") and 'review' in f]
print("review files : ")
print(review_files)
print(convert_filename(review_files[0]))
for f in review_files:
    print(f)
    temp_df = sample_json_gz_to_df(f)
    metadatafile = convert_filename(f)
    metadatadf = pd.read_json(metadatafile, lines = True)
    temp_df = temp_df.merge(metadatadf, on = 'gmap_id', how = 'left')
    temp_df.to_csv(f[:-3] + ".csv")
