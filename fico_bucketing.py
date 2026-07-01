import pandas as pd
import numpy as np

def calculate_log_likelihood(n, k):
    """
    Calculates the log-likelihood of a bucket.
    n: total number of records in the bucket
    k: number of defaults in the bucket
    """
    if n == 0:
        return 0.0
        
    p = k / n
    
    # Handle p=0 or p=1 to avoid Math Domain Error with log(0)
    # When p=0, k=0. 0*log(0) -> 0. (n-k)*log(1) -> 0.
    # When p=1, k=n. n*log(1) -> 0. (n-n)*log(0) -> 0.
    if p == 0 or p == 1:
        return 0.0
        
    return k * np.log(p) + (n - k) * np.log(1 - p)

def optimal_fico_buckets_dp(df: pd.DataFrame, num_buckets: int = 10):
    """
    Uses Dynamic Programming to find the optimal FICO score boundaries that
    maximize the log-likelihood of default distributions.
    
    Args:
        df: DataFrame containing 'fico_score' and 'default' columns.
        num_buckets: The desired number of buckets to split the FICO scores into.
        
    Returns:
        List of upper boundary FICO scores for the buckets.
    """
    # 1. Aggregate data by distinct FICO scores
    grouped = df.groupby('fico_score')['default'].agg(
        total_records='count', 
        total_defaults='sum'
    ).reset_index().sort_values('fico_score')
    
    ficos = grouped['fico_score'].values
    n = grouped['total_records'].values
    k = grouped['total_defaults'].values
    num_distinct_scores = len(ficos)
    
    # Pre-calculate prefix sums to compute range sums in O(1) time
    # prefix_n[i] will be sum of n from 0 to i-1
    prefix_n = np.zeros(num_distinct_scores + 1, dtype=int)
    prefix_k = np.zeros(num_distinct_scores + 1, dtype=int)
    for i in range(num_distinct_scores):
        prefix_n[i+1] = prefix_n[i] + n[i]
        prefix_k[i+1] = prefix_k[i] + k[i]
        
    # Memoization cache for log likelihoods of ranges (a to b inclusive)
    ll_memo = {}
    
    def range_ll(a, b):
        if (a, b) in ll_memo:
            return ll_memo[(a, b)]
            
        tot_n = prefix_n[b + 1] - prefix_n[a]
        tot_k = prefix_k[b + 1] - prefix_k[a]
        
        val = calculate_log_likelihood(tot_n, tot_k)
        ll_memo[(a, b)] = val
        return val

    # 2. Dynamic Programming setup
    # dp[b][i] = max log-likelihood for distributing first i+1 elements into b buckets
    dp = np.full((num_buckets + 1, num_distinct_scores), -np.inf)
    # split[b][i] keeps track of the optimal split point to reconstruct the boundaries
    split = np.zeros((num_buckets + 1, num_distinct_scores), dtype=int)
    
    # Base case: 1 bucket covering from index 0 to i
    for i in range(num_distinct_scores):
        dp[1][i] = range_ll(0, i)
        
    # 3. Fill the DP table
    for b in range(2, num_buckets + 1):
        for i in range(b - 1, num_distinct_scores):
            max_ll = -np.inf
            best_split = -1
            
            # Try all possible boundaries 'j' for the end of the previous bucket
            for j in range(b - 2, i):
                current_ll = dp[b - 1][j] + range_ll(j + 1, i)
                if current_ll > max_ll:
                    max_ll = current_ll
                    best_split = j
                    
            dp[b][i] = max_ll
            split[b][i] = best_split
            
    # 4. Reconstruct the boundaries
    boundaries_indices = []
    curr_end = num_distinct_scores - 1
    
    for b in range(num_buckets, 1, -1):
        j = split[b][curr_end]
        boundaries_indices.append(j)
        curr_end = j
        
    # Reverse to get them in ascending order
    boundaries_indices.reverse()
    
    # Convert indices to actual FICO scores
    boundary_ficos = [ficos[idx] for idx in boundaries_indices]
    
    return boundary_ficos

def display_bucket_stats(df, boundaries):
    """
    Helper function to print out statistics for the generated buckets.
    """
    print(f"{'FICO Range':<20} | {'Total Loans':<12} | {'Defaults':<10} | {'Probability of Default (PD)'}")
    print("-" * 75)
    
    lower_bound = 0
    for idx, upper_bound in enumerate(boundaries + [float('inf')]):
        if upper_bound == float('inf'):
            bucket_df = df[(df['fico_score'] > lower_bound)]
            range_str = f"> {lower_bound}"
        else:
            bucket_df = df[(df['fico_score'] > lower_bound) & (df['fico_score'] <= upper_bound)]
            range_str = f"{lower_bound + 1} - {upper_bound}"
            
        n = len(bucket_df)
        k = bucket_df['default'].sum()
        pd_val = (k / n * 100) if n > 0 else 0
        
        print(f"{range_str:<20} | {n:<12} | {k:<10} | {pd_val:.2f}%")
        lower_bound = upper_bound


if __name__ == "__main__":
    csv_file = 'loan_data.csv'
    
    try:
        df = pd.read_csv(csv_file)
        print("Data loaded successfully.")
        
        # User requested creating buckets, let's create 5 buckets as a good baseline
        NUM_BUCKETS = 5
        
        print(f"\nCalculating optimal boundaries for {NUM_BUCKETS} FICO buckets using Dynamic Programming...")
        optimal_boundaries = optimal_fico_buckets_dp(df, num_buckets=NUM_BUCKETS)
        
        print("\nOptimal FICO Score Boundaries (Upper Bounds):", optimal_boundaries)
        print("\nBucket Analysis:")
        display_bucket_stats(df, optimal_boundaries)
        
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'. Please ensure the file is saved in the same directory.")
