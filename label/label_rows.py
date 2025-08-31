import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import re
from openai import OpenAI

client = OpenAI(
            api_key="sk-8c02cad1fe31455ea8de173c8c87cdb9",
    base_url="https://api.deepseek.com"
)
# -------------------------
# Rule-based ultra-fast pre-check
# -------------------------
phone_regex = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\d{5,15}\b')
blacklist = ["promo code", "click here", "http", "www.", ".com", ".net", ".org", 
             "@gmail", "@yahoo", "@hotmail", "whatsapp", "telegram", "discord"]

def ultra_fast_pre_check(text):
    if not isinstance(text, str):
        return "invalid"
    
    if not text or len(text.strip()) < 20:
        return "invalid"
    
    text_lower = text.lower()
    if any(kw in text_lower for kw in blacklist):
        return "invalid"
    
    if phone_regex.search(text):
        return "invalid"
    
    words = text_lower.split()
    if len(words) > 15 and len(set(words)) / len(words) < 0.4:
        return "invalid"
    
    if len(text) > 50 and sum(c.isalpha() for c in text) / len(text) < 0.4:
        return "invalid"
    
    return "needs_ai_check"

# -------------------------
# Batch API call
# -------------------------
def generate_response_batch(df_batch):
    prompts = "\n\n".join([
        f"{i+1}. Category: {row['category']}\nText: {row['text']}" 
        for i, row in df_batch.iterrows()
    ])

    instructions = """
Classify each review as 'valid' or 'invalid' according to these rules:
- Text length under 10 words
- Excessive repetition of words and emojis
- Contains links, emails, or phone numbers
- Rating & text review mismatch
- Duplicate reviews of the same place
- Contains meaningless gibberish
- Contains spammy or promotional terms
- Location inconsistency between place and text review
- Context inconsistency between products and place
- Contains harsh words

Respond ONLY with 'valid' or 'invalid' for each numbered review, in order.
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a review quality checker."},
            {"role": "user", "content": f"{instructions}\n\n{prompts}"}
        ],
        stream=False
    )

    raw_lines = response.choices[0].message.content.strip().splitlines()

    # Normalize output: keep only 'valid'/'invalid'
    labels = [line.strip().lower() for line in raw_lines if line.strip().lower() in ("valid", "invalid")]

    # Ensure alignment with df_batch length
    if len(labels) != len(df_batch):
        print(f"⚠️ Warning: Got {len(labels)} labels for {len(df_batch)} rows. Fixing mismatch...")
        if len(labels) > len(df_batch):
            labels = labels[:len(df_batch)]
        else:
            labels.extend(["error"] * (len(df_batch) - len(labels)))

    return labels

# -------------------------
# Optional fast duplicate detection
# -------------------------
from fuzzywuzzy import fuzz

def find_duplicates_fast(texts, similarity_threshold=0.8):
    duplicates = set()
    for i, text1 in enumerate(texts):
        if i in duplicates:
            continue
        for j, text2 in enumerate(texts[i+1:], i+1):
            if j in duplicates:
                continue
            if fuzz.ratio(text1, text2) > similarity_threshold * 100:
                duplicates.add(j)
    return duplicates

# -------------------------
# Main ultra-fast pipeline
# -------------------------
def runDataframe_ultra_batch(df, batch_size=10, check_duplicates=False, similarity_threshold=0.8):
    """
    Ultra-fast dataframe processing pipeline
    """
    # Step 1: Pre-check
    with ProcessPoolExecutor() as executor:
        df['pre_check'] = list(executor.map(ultra_fast_pre_check, df['text']))
    
    non_ai_rows = df[df['pre_check'] != 'needs_ai_check'].copy()
    non_ai_rows['label'] = non_ai_rows['pre_check']
    ai_rows = df[df['pre_check'] == 'needs_ai_check'].copy()

    print(f"{len(ai_rows)} rows need AI processing out of {len(df)} total.")

    # Step 2: Batch AI processing
    if len(ai_rows) > 0:
        ai_labels = []
        for i in tqdm(range(0, len(ai_rows), batch_size), desc="AI batches"):
            batch_df = ai_rows.iloc[i:i + batch_size]  # <-- make sure this is a DataFrame
            try:
                batch_labels = generate_response_batch(batch_df)
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                batch_labels = ["error"] * len(batch_df)
            ai_labels.extend(batch_labels)
        ai_rows['label'] = ai_labels

    # Step 3: Optional duplicate detection
    if check_duplicates:
        all_texts = pd.concat([non_ai_rows, ai_rows])['text'].tolist()
        duplicates = find_duplicates_fast(all_texts, similarity_threshold)
        final_df = pd.concat([non_ai_rows, ai_rows]).reset_index(drop=True)
        final_df['label'] = final_df.apply(lambda row: "invalid" if row.name in duplicates else row['label'], axis=1)
    else:
        final_df = pd.concat([non_ai_rows, ai_rows]).sort_index()

    return final_df.drop(columns=['pre_check'], errors='ignore')

if __name__ == "__main__":
    df = pd.read_csv("combined.csv")
    new_df = runDataframe_ultra_batch(df)
    new_df.to_csv("labelled_data.csv", index = False)
    print("Successfully labelled!")
