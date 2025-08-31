#!/usr/bin/env python3
"""
Reviewer Distribution Analysis Script
Creates bar charts showing the distribution of reviews per reviewer.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

def analyze_reviewer_distribution(csv_file, output_dir="plots", author_col="author_name", user_id_col="user_id"):
    """
    Analyze and visualize the distribution of reviews per reviewer.
    
    Args:
        csv_file: Path to the CSV file containing reviews
        output_dir: Directory to save plots
        author_col: Column name for reviewer names
        user_id_col: Column name for user IDs
    """
    
    print(f"📊 Loading data from {csv_file}...")
    
    # Load the data
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df):,} reviews")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # === ANALYSIS 1: Reviews per Author Name ===
    if author_col in df.columns:
        print(f"\n🔍 Analyzing reviews per author ({author_col})...")
        
        author_counts = df[author_col].value_counts()
        print(f"📈 Found {len(author_counts):,} unique authors")
        print(f"📊 Review distribution:")
        print(f"   Mean: {author_counts.mean():.2f} reviews per author")
        print(f"   Median: {author_counts.median():.1f} reviews per author")
        print(f"   Max: {author_counts.max():,} reviews (by {author_counts.index[0]})")
        
        # Create distribution plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Full distribution
        bins = np.logspace(0, np.log10(author_counts.max()), 30)
        ax1.hist(author_counts, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xscale('log')
        ax1.set_xlabel('Reviews per Author (log scale)')
        ax1.set_ylabel('Number of Authors')
        ax1.set_title('Distribution of Reviews per Author\n(Full Range - Log Scale)')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Total Authors: {len(author_counts):,}\nMean: {author_counts.mean():.1f}\nMedian: {author_counts.median():.1f}\nMax: {author_counts.max():,}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Zoomed view (1-20 reviews)
        author_counts_limited = author_counts[author_counts <= 20]
        ax2.hist(author_counts_limited, bins=range(1, 22), alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Reviews per Author')
        ax2.set_ylabel('Number of Authors')
        ax2.set_title('Distribution of Reviews per Author\n(1-20 Reviews - Linear Scale)')
        ax2.set_xticks(range(1, 21, 2))
        ax2.grid(True, alpha=0.3)
        
        # Add percentage of authors with 1-20 reviews
        pct_limited = len(author_counts_limited) / len(author_counts) * 100
        ax2.text(0.02, 0.98, f"{pct_limited:.1f}% of authors\nhave ≤20 reviews", 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/reviewer_distribution_by_author.png", dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot: {output_dir}/reviewer_distribution_by_author.png")
        plt.show()
        
        # Show top reviewers
        print(f"\n🏆 TOP 50 MOST ACTIVE REVIEWERS:")
        top_reviewers = []
        for i, (author, count) in enumerate(author_counts.head(50).items(), 1):
            print(f"  {i:2d}. {author:<40} {count:,} reviews")
            top_reviewers.append({'rank': i, 'author': author, 'review_count': count})
        
        # Save top reviewers to CSV
        top_reviewers_df = pd.DataFrame(top_reviewers)
        top_reviewers_csv = f"{output_dir}/top_50_reviewers_by_author.csv"
        top_reviewers_df.to_csv(top_reviewers_csv, index=False)
        print(f"💾 Saved top reviewers: {top_reviewers_csv}")
    
    # === ANALYSIS 2: Reviews per User ID ===
    if user_id_col in df.columns:
        print(f"\n🔍 Analyzing reviews per user ID ({user_id_col})...")
        
        user_counts = df[user_id_col].value_counts()
        print(f"📈 Found {len(user_counts):,} unique user IDs")
        print(f"📊 Review distribution:")
        print(f"   Mean: {user_counts.mean():.2f} reviews per user")
        print(f"   Median: {user_counts.median():.1f} reviews per user")
        print(f"   Max: {user_counts.max():,} reviews")
        
        # Create distribution plot for user IDs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Full distribution
        bins = np.logspace(0, np.log10(user_counts.max()), 30)
        ax1.hist(user_counts, bins=bins, alpha=0.7, color='lightgreen', edgecolor='black')
        ax1.set_xscale('log')
        ax1.set_xlabel('Reviews per User ID (log scale)')
        ax1.set_ylabel('Number of Users')
        ax1.set_title('Distribution of Reviews per User ID\n(Full Range - Log Scale)')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Total Users: {len(user_counts):,}\nMean: {user_counts.mean():.1f}\nMedian: {user_counts.median():.1f}\nMax: {user_counts.max():,}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Zoomed view (1-20 reviews)
        user_counts_limited = user_counts[user_counts <= 20]
        ax2.hist(user_counts_limited, bins=range(1, 22), alpha=0.7, color='gold', edgecolor='black')
        ax2.set_xlabel('Reviews per User ID')
        ax2.set_ylabel('Number of Users')
        ax2.set_title('Distribution of Reviews per User ID\n(1-20 Reviews - Linear Scale)')
        ax2.set_xticks(range(1, 21, 2))
        ax2.grid(True, alpha=0.3)
        
        # Add percentage
        pct_limited = len(user_counts_limited) / len(user_counts) * 100
        ax2.text(0.02, 0.98, f"{pct_limited:.1f}% of users\nhave ≤20 reviews", 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/reviewer_distribution_by_user_id.png", dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot: {output_dir}/reviewer_distribution_by_user_id.png")
        plt.show()
        
        # Show top users by review count
        print(f"\n🏆 TOP 50 MOST ACTIVE USER IDs:")
        top_users = []
        for i, (user_id, count) in enumerate(user_counts.head(50).items(), 1):
            print(f"  {i:2d}. {user_id:<40} {count:,} reviews")
            top_users.append({'rank': i, 'user_id': user_id, 'review_count': count})
        
        # Save top users to CSV
        top_users_df = pd.DataFrame(top_users)
        top_users_csv = f"{output_dir}/top_50_reviewers_by_user_id.csv"
        top_users_df.to_csv(top_users_csv, index=False)
        print(f"💾 Saved top users: {top_users_csv}")
    
    # === ANALYSIS 3: Comparison and Insights ===
    if author_col in df.columns and user_id_col in df.columns:
        print(f"\n🤔 COMPARISON INSIGHTS:")
        
        # Check for mismatches (same user_id with different names)
        user_to_names = df.groupby(user_id_col)[author_col].nunique()
        multiple_names = user_to_names[user_to_names > 1]
        
        if len(multiple_names) > 0:
            print(f"⚠️  Found {len(multiple_names):,} user IDs with multiple author names")
            print(f"   This suggests some users might have changed names or data inconsistencies")
            
            # Show examples
            print(f"\n📝 Examples of users with multiple names:")
            for user_id in multiple_names.head(5).index:
                names = df[df[user_id_col] == user_id][author_col].unique()
                print(f"   User {user_id}: {list(names)}")
        
        # Check for name collisions (same name with different user_ids)
        name_to_users = df.groupby(author_col)[user_id_col].nunique()
        multiple_users = name_to_users[name_to_users > 1]
        
        if len(multiple_users) > 0:
            print(f"⚠️  Found {len(multiple_users):,} author names shared by multiple user IDs")
            print(f"   This suggests common names or potential duplicate accounts")
    
    # === ANALYSIS 4: Power Law Analysis ===
    print(f"\n⚡ POWER LAW ANALYSIS:")
    if author_col in df.columns:
        author_counts = df[author_col].value_counts()
        
        # Calculate percentiles
        total_reviews = len(df)
        top_1_pct_authors = max(1, len(author_counts) // 100)
        top_5_pct_authors = max(1, len(author_counts) // 20)
        top_10_pct_authors = max(1, len(author_counts) // 10)
        
        reviews_by_top_1_pct = author_counts.head(top_1_pct_authors).sum()
        reviews_by_top_5_pct = author_counts.head(top_5_pct_authors).sum()
        reviews_by_top_10_pct = author_counts.head(top_10_pct_authors).sum()
        
        print(f"📊 Review concentration:")
        print(f"   Top 1% of authors ({top_1_pct_authors:,}) wrote {reviews_by_top_1_pct:,} reviews ({reviews_by_top_1_pct/total_reviews*100:.1f}%)")
        print(f"   Top 5% of authors ({top_5_pct_authors:,}) wrote {reviews_by_top_5_pct:,} reviews ({reviews_by_top_5_pct/total_reviews*100:.1f}%)")
        print(f"   Top 10% of authors ({top_10_pct_authors:,}) wrote {reviews_by_top_10_pct:,} reviews ({reviews_by_top_10_pct/total_reviews*100:.1f}%)")
        
        # Single reviewer analysis
        single_review_authors = (author_counts == 1).sum()
        print(f"   {single_review_authors:,} authors ({single_review_authors/len(author_counts)*100:.1f}%) wrote exactly 1 review")

    # === ANALYSIS 5: Suspicious Activity Detection ===
    print(f"\n🕵️ SUSPICIOUS ACTIVITY DETECTION:")
    if author_col in df.columns:
        author_counts = df[author_col].value_counts()
        
        # Find power reviewers (potential spam/bots)
        power_threshold = author_counts.quantile(0.99)  # Top 1%
        power_reviewers = author_counts[author_counts >= power_threshold]
        
        if len(power_reviewers) > 0:
            print(f"🚨 Found {len(power_reviewers)} potential power reviewers (≥{power_threshold:.0f} reviews):")
            for author, count in power_reviewers.head(5).items():
                print(f"   {author}: {count} reviews")
            
            if len(power_reviewers) > 5:
                print(f"   ... and {len(power_reviewers) - 5} more")
        
        # Find prolific single-day reviewers
        if 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'], unit='ms').dt.date
            daily_review_counts = df.groupby([author_col, 'date']).size()
            high_daily_activity = daily_review_counts[daily_review_counts >= 10]
            
            if len(high_daily_activity) > 0:
                print(f"\n📅 Found {len(high_daily_activity)} instances of ≥10 reviews per day:")
                for (author, date), count in high_daily_activity.head(5).items():
                    print(f"   {author} on {date}: {count} reviews")

def main():
    parser = argparse.ArgumentParser(description="Analyze reviewer distribution from CSV data")
    parser.add_argument("csv_file", help="Path to CSV file containing review data")
    parser.add_argument("--output-dir", default="plots", help="Directory to save plots (default: plots)")
    parser.add_argument("--author-col", default="author_name", help="Column name for author names")
    parser.add_argument("--user-col", default="user_id", help="Column name for user IDs")
    
    args = parser.parse_args()
    
    print("🎨 REVIEWER DISTRIBUTION ANALYZER")
    print("=" * 50)
    
    analyze_reviewer_distribution(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        author_col=args.author_col,
        user_id_col=args.user_col
    )
    
    print(f"\n✅ Analysis complete! Check the '{args.output_dir}' directory for plots.")

if __name__ == "__main__":
    main()
