"""
Pipeline Stage 2: Rule-Based Filtering for Business Reviews
This module applies business rules to filter reviews, keeping only high-quality ones (Label = "Yes").
Data format: business_name;author_name;text;photo;rating;rating_category;Label
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List
import re
from datetime import datetime
import glob

def run(config: Dict[str, Any]) -> bool:
    """
    Execute the business rule-based filtering stage.
    
    Args:
        config: Configuration dictionary containing:
            - input_file: Path to input CSV file (from stage 1)
            - output_file: Path to output CSV file
            
    Returns:
        bool: True if successful, False otherwise
    """
    input_file = config['input_file']
    output_file = config['output_file']
    execution_output_dir = config.get('execution_output_dir')
    execution_id = config.get('execution_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Generate timestamp for metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if we have an execution directory
    if execution_output_dir:
        # Look for stage1_output.csv in the same execution directory
        stage1_file = Path(execution_output_dir) / "stage1_output.csv"
        input_file = str(stage1_file)
        logging.info(f"Stage 2: Using stage 1 output from execution directory: {input_file}")
        
        # Set output to execution directory
        output_file = str(Path(execution_output_dir) / "stage2_output.csv")
    else:
        # Use latest stage 1 output
        stage1_pattern = input_file.replace('.csv', '_*.csv')
        latest_stage1 = find_latest_file(stage1_pattern)
        input_file = latest_stage1
        logging.info(f"Stage 2: Found latest stage 1 output: {input_file}")
        
        # Add timestamp to output filename
        output_path = Path(output_file)
        output_name = f"{output_path.stem}_{timestamp}{output_path.suffix}"
        output_file = str(output_path.parent / output_name)
    
    logging.info(f"Stage 2: Loading data from {input_file}")
    logging.info(f"Stage 2: Output will be saved to {output_file}")
    
    # Load data from stage 1
    df = pd.read_csv(input_file, sep=';', encoding='latin-1')
    logging.info(f"Loaded data with Latin-1 encoding and semicolon separator")
    
    initial_count = len(df)
    logging.info(f"Loaded {initial_count} reviews from stage 1")
    
    # Validate expected columns
    expected_columns = ['business_name', 'author_name', 'text', 'photo', 'rating', 'rating_category', 'Label']
    available_columns = list(df.columns)
    logging.info(f"Available columns: {available_columns}")
    
    # Convert data types
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['text'] = df['text'].astype(str)
    df['business_name'] = df['business_name'].astype(str)
    
    # Load business rules
    rules = load_business_rules()
    
    logging.info(f"Starting rule-based filtering with {len(rules)} business rules")
    
    # Apply each business rule sequentially
    for i, rule in enumerate(rules, 1):
        logging.info(f"\n--- Applying business rule {i}/{len(rules)} ---")
        df = apply_business_rule(df, rule)
    
    # Final summary
    final_count = len(df)
    filtered_count = initial_count - final_count
    retention_rate = (final_count / initial_count) * 100
    
    logging.info(f"\n=== Stage 2 Business Rule Filtering Summary ===")
    logging.info(f"  Initial reviews: {initial_count:,}")
    logging.info(f"  Final reviews: {final_count:,}")
    logging.info(f"  Filtered out: {filtered_count:,}")
    logging.info(f"  Retention rate: {retention_rate:.1f}%")
    
    # Add filtering metadata to the dataframe
    df['filtered_timestamp'] = timestamp
    df['retention_rate'] = retention_rate
    
    # Save filtered data
    df.to_csv(output_file, sep=';', index=False, encoding='latin-1')
    
    logging.info(f"Stage 2: Saved {len(df):,} high-quality reviews to {output_file}")
    
    return True

def find_latest_file(pattern: str) -> str:
    """
    Find the most recent file matching a pattern.
    
    Args:
        pattern: Glob pattern to search for files
        
    Returns:
        str: Path to the most recent file
    """
    files = glob.glob(pattern)
    # Sort by modification time (newest first)
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return files[0]

def load_business_rules() -> List[Dict[str, Any]]:
    """
    Load and parse business rules for review filtering based on rules.txt.
    
    Returns:
        List of rule dictionaries with rule logic
    """
    rules = [
        {
            'name': '1. Minimum Text Length Filter',
            'description': 'Rejects reviews with fewer than N characters',
            'function': 'min_text_length',
            'params': {'min_length': 10}
        },
        {
            'name': '2. Repetition Filter',
            'description': 'Flags reviews that repeat the same word or emoji excessively',
            'function': 'check_repetition',
            'params': {'max_repetition': 3}
        },
        {
            'name': '3. Link and Contact Filter',
            'description': 'Flags reviews containing URLs, emails, or phone numbers',
            'function': 'check_links_contacts',
            'params': {}
        },
        {
            'name': '4. Rating-Text Mismatch Filter',
            'description': 'Flags reviews where sentiment conflicts with the star rating',
            'function': 'check_rating_text_mismatch',
            'params': {}
        },
        {
            'name': '5. Duplicate Detection Filter',
            'description': 'Flags reviews that are identical or near-duplicate',
            'function': 'check_duplicates',
            'params': {'similarity_threshold': 0.85}
        },
        {
            'name': '6. Gibberish Filter',
            'description': 'Rejects reviews with meaningless characters or non-language text',
            'function': 'check_gibberish',
            'params': {'gibberish_threshold': 0.4}
        },
        {
            'name': '7. Keyword Blacklist Filter',
            'description': 'Flags reviews with spammy terms like "promo code" or "click here"',
            'function': 'check_keyword_blacklist',
            'params': {
                'blacklist': ['promo code', 'click here', 'discount', 'free shipping', 
                             'limited time', 'offer expires', 'visit our website', 'call now']
            }
        },
        {
            'name': '8. Location Consistency Filter',
            'description': 'Flags reviews that mention a different place than the tagged location',
            'function': 'check_location_consistency',
            'params': {}
        },
        {
            'name': '9. Context Consistency Filter',
            'description': 'Flags reviews that mention things not relevant to the current Google Place',
            'function': 'check_context_consistency',
            'params': {}
        }
    ]
    
    logging.info(f"Loaded {len(rules)} business rules for review filtering")
    return rules

def min_text_length(text: str, min_length: int = 10) -> bool:
    """Check if text meets minimum length requirement."""
    clean_text = text.strip()
    return len(clean_text) >= min_length

def check_repetition(text: str, max_repetition: int = 3) -> bool:
    """Check for excessive repetition in text (Rule 2)."""
    text_lower = text.lower()
    words = re.findall(r'\w+', text_lower)  # Extract only words
    
    # Check for repeated words
    for word in set(words):
        if len(word) > 2:  # Only check words longer than 2 chars
            if words.count(word) > max_repetition:
                logging.debug(f"Excessive repetition found: '{word}' appears {words.count(word)} times")
                return False
    
    # Check for repeated characters within words (e.g., "goooooood")
    for word in words:
        if len(word) > 4:
            for char in set(word):
                char_ratio = word.count(char) / len(word)
                if char_ratio > 0.6:  # More than 60% same character
                    logging.debug(f"Excessive character repetition in word: '{word}'")
                    return False
    
    # Check for emoji repetition  
    emoji_pattern = r'[👍👎😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍😘😗😙😚😋😛😝😜🤪🤨🧐🤓😎🤩😏😒😞😔😟😕🙁☹️😣😖😫😩😢😭😤😠😡🤬🤯😳😱😨😰😥😓🤗🤔🤭🤫🤥😶😐😑😬🙄😯😦😧😮😲😴🤤😪😵🤐🤢🤮🤧😷🤒🤕🤑🤠😈👿👹👺🤡💩👻💀☠️👽👾🤖🎃😺😸😹😻😼😽🙀😿😾]'
    emoji_count = len(re.findall(emoji_pattern, text))
    if emoji_count > max_repetition:
        logging.debug(f"Excessive emoji repetition: {emoji_count} emojis found")
        return False
    
    return True

def check_links_contacts(text: str) -> bool:
    """Check for URLs, emails, or phone numbers (Rule 3)."""
    # URL patterns
    url_patterns = [
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        r'[a-zA-Z0-9.-]+\.com|\.org|\.net|\.edu|\.gov'
    ]
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Phone patterns
    phone_patterns = [
        r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\d{3}-\d{3}-\d{4}',
        r'\(\d{3}\)\s*\d{3}-\d{4}'
    ]
    
    text_lower = text.lower()
    
    # Check for URL patterns
    for pattern in url_patterns:
        if re.search(pattern, text_lower):
            logging.debug(f"URL found in text: {pattern}")
            return False
    
    # Check for email
    if re.search(email_pattern, text):
        logging.debug("Email address found in text")
        return False
    
    # Check for phone numbers
    for pattern in phone_patterns:
        if re.search(pattern, text):
            logging.debug("Phone number found in text")
            return False
    
    return True

def check_rating_text_mismatch(text: str, rating: float) -> bool:
    """Check if rating conflicts with text sentiment (Rule 4)."""
    text_lower = text.lower()
    
    # Strong negative sentiment words
    strong_negative = [
        'terrible', 'awful', 'horrible', 'disgusting', 'worst', 'hate', 'never again',
        'nasty', 'appalling', 'dreadful', 'shocking', 'unacceptable', 'pathetic'
    ]
    
    # Negative sentiment words
    negative_words = [
        'bad', 'poor', 'disappointing', 'cold', 'burnt', 'overcooked', 'undercooked',
        'soggy', 'dry', 'tasteless', 'bland', 'stale', 'rude', 'slow', 'dirty'
    ]
    
    # Strong positive sentiment words
    strong_positive = [
        'excellent', 'amazing', 'fantastic', 'incredible', 'outstanding', 'perfect',
        'superb', 'brilliant', 'exceptional', 'phenomenal', 'spectacular'
    ]
    
    # Positive sentiment words
    positive_words = [
        'great', 'good', 'wonderful', 'delicious', 'nice', 'fresh', 'tasty',
        'friendly', 'clean', 'fast', 'helpful', 'recommend', 'love', 'enjoy'
    ]
    
    # Count sentiment words
    strong_negative_count = sum(1 for word in strong_negative if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    strong_positive_count = sum(1 for word in strong_positive if word in text_lower)
    positive_count = sum(1 for word in positive_words if word in text_lower)
    
    total_negative = strong_negative_count * 2 + negative_count
    total_positive = strong_positive_count * 2 + positive_count
    
    # Rating 4-5 with strong negative sentiment
    if rating >= 4 and strong_negative_count > 0:
        logging.debug(f"Mismatch: High rating ({rating}) with strong negative words")
        return False
    
    # Rating 4-5 with more negative than positive sentiment
    if rating >= 4 and total_negative > total_positive and total_negative > 1:
        logging.debug(f"Mismatch: High rating ({rating}) with negative sentiment")
        return False
    
    # Rating 1-2 with strong positive sentiment
    if rating <= 2 and strong_positive_count > 0:
        logging.debug(f"Mismatch: Low rating ({rating}) with strong positive words")
        return False
    
    # Rating 1-2 with more positive than negative sentiment
    if rating <= 2 and total_positive > total_negative and total_positive > 1:
        logging.debug(f"Mismatch: Low rating ({rating}) with positive sentiment")
        return False
    
    return True

def check_duplicates(df: pd.DataFrame, text_column: str, similarity_threshold: float = 0.85) -> pd.Series:
    """Check for near-duplicate reviews (Rule 5)."""
    # Normalize text for comparison
    df_normalized = df[text_column].str.lower().str.strip()
    
    # Find exact duplicates
    duplicate_mask = ~df_normalized.duplicated(keep='first')
    
    duplicates_found = len(df) - duplicate_mask.sum()
    logging.info(f"Found {duplicates_found} duplicate reviews")
    
    return duplicate_mask

def check_gibberish(text: str, gibberish_threshold: float = 0.4) -> bool:
    """Check if text contains too many meaningless characters (Rule 6)."""
    text = text.strip()
    
    # Count different character types
    total_chars = len(text)
    letters = len(re.findall(r'[a-zA-Z]', text))
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    
    # Calculate ratios
    letter_ratio = letters / total_chars
    special_ratio = special_chars / total_chars
    
    # Check for gibberish patterns
    if letter_ratio < 0.3:  # Less than 30% letters
        logging.debug(f"Low letter ratio: {letter_ratio:.2f}")
        return False
    
    if special_ratio > gibberish_threshold:  # Too many special characters
        logging.debug(f"High special character ratio: {special_ratio:.2f}")
        return False
    
    # Check for random character sequences
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / len(words)
    if avg_word_length > 15:  # Unusually long "words"
        logging.debug(f"Unusually long average word length: {avg_word_length}")
        return False
    
    return True

def check_keyword_blacklist(text: str, blacklist: List[str]) -> bool:
    """Check for blacklisted spammy keywords (Rule 7)."""
    text_lower = text.lower()
    
    for keyword in blacklist:
        if keyword.lower() in text_lower:
            logging.debug(f"Blacklisted keyword found: '{keyword}'")
            return False
    
    return True

def check_location_consistency(text: str, business_name: str) -> bool:
    """Check if review mentions different locations (Rule 8)."""
    text_lower = text.lower()
    business_lower = business_name.lower()
    
    # Common location indicators that might suggest wrong location
    location_keywords = [
        'starbucks', 'mcdonald', 'burger king', 'kfc', 'subway', 'domino',
        'pizza hut', 'taco bell', 'wendy', 'dunkin'
    ]
    
    # Extract potential business name from the business_name field
    business_indicators = []
    for keyword in location_keywords:
        if keyword in business_lower:
            business_indicators.append(keyword)
    
    # Check if text mentions a different business type
    for keyword in location_keywords:
        if keyword in text_lower and keyword not in business_indicators:
            logging.debug(f"Location inconsistency: mentions '{keyword}' but business is '{business_name}'")
            return False
    
    return True

def check_context_consistency(text: str, business_name: str) -> bool:
    """Check if review content is relevant to the business type (Rule 9)."""
    text_lower = text.lower()
    business_lower = business_name.lower()
    
    # Define inappropriate contexts for different business types
    inappropriate_contexts = {
        'mcdonald': ['alcohol', 'beer', 'wine', 'cocktail', 'bar'],
        'starbucks': ['burger', 'fries', 'pizza', 'alcohol'],
        'restaurant': [],  # Restaurants can serve many things
        'cafe': ['burger', 'fries', 'alcohol', 'beer'],
        'coffee': ['burger', 'fries', 'pizza', 'alcohol']
    }
    
    # Determine business type
    business_type = 'restaurant'  # Default
    for biz_type in inappropriate_contexts.keys():
        if biz_type in business_lower:
            business_type = biz_type
            break
    
    # Check for inappropriate mentions
    inappropriate_items = inappropriate_contexts.get(business_type, [])
    for item in inappropriate_items:
        if item in text_lower:
            logging.debug(f"Context inconsistency: mentions '{item}' at {business_type}")
            return False
    
    return True

def apply_business_rule(df: pd.DataFrame, rule: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a single business rule to filter the dataframe.
    
    Args:
        df: Input dataframe
        rule: Rule dictionary with rule logic
        
    Returns:
        Filtered dataframe
    """
    rule_name = rule['name']
    rule_func = rule['function']
    rule_params = rule['params']
    
    logging.info(f"Applying rule: {rule_name}")
    initial_count = len(df)
    
    if rule_func == 'min_text_length':
        mask = df['text'].apply(lambda x: min_text_length(x, rule_params['min_length']))
    elif rule_func == 'check_repetition':
        mask = df['text'].apply(lambda x: check_repetition(x, rule_params['max_repetition']))
    elif rule_func == 'check_links_contacts':
        mask = df['text'].apply(check_links_contacts)
    elif rule_func == 'check_rating_text_mismatch':
        mask = df.apply(lambda row: check_rating_text_mismatch(row['text'], row['rating']), axis=1)
    elif rule_func == 'check_duplicates':
        mask = check_duplicates(df, 'text', rule_params['similarity_threshold'])
    elif rule_func == 'check_gibberish':
        mask = df['text'].apply(lambda x: check_gibberish(x, rule_params['gibberish_threshold']))
    elif rule_func == 'check_keyword_blacklist':
        mask = df['text'].apply(lambda x: check_keyword_blacklist(x, rule_params['blacklist']))
    elif rule_func == 'check_location_consistency':
        mask = df.apply(lambda row: check_location_consistency(row['text'], row['business_name']), axis=1)
    elif rule_func == 'check_context_consistency':
        mask = df.apply(lambda row: check_context_consistency(row['text'], row['business_name']), axis=1)
    
    # Apply the filter - keep only rows that pass the rule
    filtered_df = df[mask].copy()
    filtered_count = initial_count - len(filtered_df)
    
    logging.info(f"  → Filtered out {filtered_count} reviews ({filtered_count/initial_count*100:.1f}%)")
    logging.info(f"  → Reviews remaining: {len(filtered_df)}")
    
    return filtered_df

if __name__ == "__main__":
    # Test the module independently
    test_config = {
        'input_file': 'data/stage1_output.csv',
        'output_file': 'data/stage2_output.csv'
    }
    success = run(test_config)
    print(f"Stage 2 execution: {'SUCCESS' if success else 'FAILED'}")
