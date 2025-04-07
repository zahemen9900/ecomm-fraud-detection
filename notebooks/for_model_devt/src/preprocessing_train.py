"""
Optimized data preprocessing module for e-commerce fraud detection.
This version includes performance optimizations for address handling.
"""
import sys
import pandas as pd
import numpy as np
import random
from datetime import datetime
from typing import Tuple, Optional, List, Dict
import multiprocessing
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import OrdinalEncoder
import pickle
import json
import hashlib
from pathlib import Path

# Import original functions to reuse
from .preprocessing import (
    load_data, merge_datasets, clean_data, generate_ghanaian_location,
    generate_ghanaian_address, _process_time_features, _one_hot_encode_chunk,
    prepare_data_for_modeling
)

# Try to import tqdm, use a simple pass-through if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class AddressPool:
    """Class to manage pools of pre-generated addresses and locations."""
    
    def __init__(self, pool_size=10000, cache_dir=None):
        """
        Initialize the address pool.
        
        Args:
            pool_size: Number of addresses to pre-generate
            cache_dir: Directory to store/load cached address pools
        """
        self.pool_size = pool_size
        self.cache_dir = cache_dir if cache_dir else Path.home() / '.cache' / 'ecomm-fraud'
        self.locations = []
        self.addresses = []
        self.initialized = False
        
        # Create cache directory if it doesn't exist
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_cache_filename(self, prefix='address_pool'):
        """Generate a deterministic cache filename based on pool size."""
        return self.cache_dir / f"{prefix}_{self.pool_size}.pkl"
    
    def initialize(self, force_regenerate=False):
        """
        Initialize the pools, either by loading from cache or generating new data.
        
        Args:
            force_regenerate: If True, regenerate pools even if cache exists
        """
        if self.initialized and not force_regenerate:
            return
            
        location_cache = self._generate_cache_filename('location_pool')
        address_cache = self._generate_cache_filename('address_pool')
        
        # Try to load from cache
        if not force_regenerate:
            try:
                if location_cache.exists() and address_cache.exists():
                    print(f"Loading address pools from cache ({self.pool_size} entries)...")
                    with open(location_cache, 'rb') as f:
                        self.locations = pickle.load(f)
                    with open(address_cache, 'rb') as f:
                        self.addresses = pickle.load(f)
                    self.initialized = True
                    return
            except Exception as e:
                print(f"Error loading from cache: {e}. Regenerating...")
        
        # Generate new pools
        print(f"Generating address pools ({self.pool_size} entries)...")
        
        # Use parallel generation for large pools
        if self.pool_size > 1000:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
            chunk_size = min(5000, max(100, self.pool_size // n_jobs))
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Generate locations
                location_futures = [
                    executor.submit(self._generate_locations_chunk, chunk_size) 
                    for _ in range((self.pool_size + chunk_size - 1) // chunk_size)
                ]
                
                # Generate addresses
                address_futures = [
                    executor.submit(self._generate_addresses_chunk, chunk_size) 
                    for _ in range((self.pool_size + chunk_size - 1) // chunk_size)
                ]
                
                # Collect results
                for future in tqdm(as_completed(location_futures), 
                                  total=len(location_futures), 
                                  desc="Generating locations"):
                    self.locations.extend(future.result())
                
                for future in tqdm(as_completed(address_futures), 
                                  total=len(address_futures), 
                                  desc="Generating addresses"):
                    self.addresses.extend(future.result())
        else:
            # For small pools, generate directly
            self.locations = [generate_ghanaian_location()[0] for _ in range(self.pool_size)]
            self.addresses = [generate_ghanaian_address() for _ in range(self.pool_size)]
        
        # Trim to exact size
        self.locations = self.locations[:self.pool_size]
        self.addresses = self.addresses[:self.pool_size]
        
        # Cache the results
        if self.cache_dir:
            print("Saving address pools to cache...")
            with open(location_cache, 'wb') as f:
                pickle.dump(self.locations, f)
            with open(address_cache, 'wb') as f:
                pickle.dump(self.addresses, f)
                
        self.initialized = True
    
    def _generate_locations_chunk(self, size):
        """Generate a chunk of locations."""
        return [generate_ghanaian_location()[0] for _ in range(size)]
    
    def _generate_addresses_chunk(self, size):
        """Generate a chunk of addresses."""
        return [generate_ghanaian_address() for _ in range(size)]
    
    def get_location(self, index=None):
        """
        Get a location from the pool.
        
        Args:
            index: If provided, use this index, otherwise use a random index
        
        Returns:
            A location string
        """
        if not self.initialized:
            self.initialize()
            
        if index is None:
            index = random.randint(0, self.pool_size - 1)
        else:
            index = index % self.pool_size
            
        return self.locations[index]
    
    def get_address(self, index=None):
        """
        Get an address from the pool.
        
        Args:
            index: If provided, use this index, otherwise use a random index
        
        Returns:
            An address string
        """
        if not self.initialized:
            self.initialize()
            
        if index is None:
            index = random.randint(0, self.pool_size - 1)
        else:
            index = index % self.pool_size
            
        return self.addresses[index]


# Create global instance of address pool
address_pool = AddressPool(pool_size=50000)  # We'll use a large pool to reduce repetition


def transform_to_ghanaian_addresses_optimized(df):
    """
    Transform addresses in the dataset to Ghanaian addresses using pre-generated pools.
    
    Args:
        df: DataFrame containing 'Shipping Address', 'Billing Address', and 'Customer Location' columns
        
    Returns:
        DataFrame with transformed addresses
    """
    # Create copy to avoid modifying original
    df_transformed = df.copy()
    n_transactions = len(df)
    
    start_time = time.time()
    print(f"Transforming {n_transactions} addresses to Ghanaian format (optimized)...")
    
    # Initialize the address pool if needed
    if not address_pool.initialized:
        address_pool.initialize()
    
    # Generate indices for addresses - we'll ensure that same index produces same address
    # But we'll hash the original values to get deterministic but varied indices
    shipping_addr_hashes = []
    
    # First, produce hashes for all addresses
    for i in range(n_transactions):
        # Use hash of original shipping address to get a reproducible index
        addr_hash = int(hashlib.md5(str(df.iloc[i]['Shipping Address']).encode()).hexdigest(), 16)
        shipping_addr_hashes.append(addr_hash % address_pool.pool_size)
    
    # Now, use the indices to get addresses from the pool
    for i in tqdm(range(n_transactions), desc="Transforming addresses"):
        # Get the indexed address
        addr_idx = shipping_addr_hashes[i]
        location_idx = (addr_idx * 17) % address_pool.pool_size  # Use a different index for location
        
        # Transform Customer Location
        df_transformed.loc[i, 'Customer Location'] = address_pool.get_location(location_idx)
        
        if 'Same_Address' in df_transformed.columns:
            same_address = df_transformed.loc[i, 'Same_Address'] == 1
        else:
            # Infer from matching addresses
            same_address = df_transformed.loc[i, 'Shipping Address'] == df_transformed.loc[i, 'Billing Address']
            
        # Get shipping address from pool
        shipping_addr = address_pool.get_address(addr_idx)
        df_transformed.loc[i, 'Shipping Address'] = shipping_addr
            
        if same_address:
            # Use same address for billing
            df_transformed.loc[i, 'Billing Address'] = shipping_addr
        else:
            # Use a different but deterministic address for billing
            bill_idx = (addr_idx * 31 + 17) % address_pool.pool_size  # Simple hash function
            df_transformed.loc[i, 'Billing Address'] = address_pool.get_address(bill_idx)
    
    print(f"Address transformation completed in {time.time() - start_time:.2f} seconds")
    return df_transformed


def engineer_features_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on transaction data with optimized address transformation.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with additional engineered features
    """
    start_time = time.time()
    print(f"Engineering features for {len(df)} records (optimized version)...")
    
    # Create a copy to avoid modifying the original
    enhanced_df = df.copy()
    
    # Compute amount per item (vectorized operation - already fast)
    enhanced_df['Amount_per_Item'] = enhanced_df['Transaction Amount'] / enhanced_df['Quantity']
    
    # Shipping-billing address match (vectorized operation - already fast)
    enhanced_df['Same_Address'] = (enhanced_df['Shipping Address'] == 
                                  enhanced_df['Billing Address']).astype(int)
    
    # For time-based features, use parallel processing if dataset is large
    if len(df) > 10000:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        batch_size = min(5000, max(1000, len(df) // (n_jobs * 2)))
        
        # Split into batches
        chunks = [enhanced_df.iloc[i:i+batch_size].copy() for i in range(0, len(enhanced_df), batch_size)]
        print(f"Processing time features in {len(chunks)} parallel batches...")
        
        # Process time features in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            processed_chunks = list(tqdm(
                executor.map(_process_time_features, chunks), 
                total=len(chunks),
                desc="Engineering time features"
            ))
        
        # Combine results
        enhanced_df = pd.concat(processed_chunks)
        
        # Make sure the original order is preserved
        enhanced_df = enhanced_df.sort_index()
    else:
        # For smaller datasets, process time features directly
        print("Processing time features...")
        enhanced_df = _process_time_features(enhanced_df)
    
    # Transform addresses using the optimized method
    print("Transforming addresses using optimized method...")
    enhanced_df = transform_to_ghanaian_addresses_optimized(enhanced_df)
    
    print(f"Feature engineering (optimized) completed in {time.time() - start_time:.2f} seconds")
    return enhanced_df


def process_data_pipeline_optimized(file_paths: List[str], warmup_pool=True, subset_percentage=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Full data processing pipeline from loading to train-test split with optimized address handling.
    
    Args:
        file_paths: List of paths to CSV files
        warmup_pool: Whether to initialize the address pool upfront
        subset_percentage: If provided, use only this percentage of the dataset (maintaining class balance)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    total_start_time = time.time()
    print(f"\nStarting optimized data processing pipeline...")
    
    # Initialize address pool first if requested
    if warmup_pool:
        print("Initializing address pools...")
        address_pool.initialize()
    
    # Load and merge data
    print("\nSTEP 1: Loading and merging data...")
    df = merge_datasets(file_paths)
    print(f"Loaded dataset with {len(df)} records and {df.shape[1]} columns")
    
    # Take a stratified subset if requested
    if subset_percentage is not None and 0 < subset_percentage < 100:
        print(f"\nTaking a {subset_percentage}% stratified subset of the data...")
        # Ensure we have the target variable
        if 'Is Fraudulent' not in df.columns:
            raise ValueError("Dataset must contain 'Is Fraudulent' column to create a stratified subset")
        
        # Calculate original class distribution
        fraud_count = df['Is Fraudulent'].sum()
        total_count = len(df)
        fraud_percentage = fraud_count / total_count * 100
        print(f"Original dataset: {total_count} samples, {fraud_count} fraudulent ({fraud_percentage:.2f}%)")
        
        # Calculate number of samples to extract
        subset_size = int(len(df) * subset_percentage / 100)
        
        # Create a stratified subset
        df_subset, _ = train_test_split(
            df, 
            test_size=(1 - subset_percentage/100),
            random_state=42,
            stratify=df['Is Fraudulent']
        )
        
        # Verify the subset class distribution
        subset_fraud_count = df_subset['Is Fraudulent'].sum()
        subset_percentage_fraud = subset_fraud_count / len(df_subset) * 100
        print(f"Subset: {len(df_subset)} samples, {subset_fraud_count} fraudulent ({subset_percentage_fraud:.2f}%)")
        
        # Replace original dataframe with subset
        df = df_subset
    
    # Clean data
    print("\nSTEP 2: Cleaning data...")
    df_cleaned = clean_data(df)
    
    # Engineer features with optimized address handling
    print("\nSTEP 3: Engineering features (optimized)...")
    df_engineered = engineer_features_optimized(df_cleaned)
    
    # Prepare for modeling
    print("\nSTEP 4: Preparing data for modeling...")
    X, y = prepare_data_for_modeling(df_engineered)
    print(f"Final processed dataset has {X.shape[1]} features after encoding")
    
    # Split into training and testing sets
    print("\nSTEP 5: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nEntire optimized data pipeline completed in {time.time() - total_start_time:.2f} seconds")
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage as a script
    import argparse
    
    parser = argparse.ArgumentParser(description='Process e-commerce data for fraud detection modeling (optimized version)')
    parser.add_argument('file_paths', nargs='+', help='Paths to CSV files to process')
    parser.add_argument('--output', type=str, help='Path to save processed data (optional)')
    parser.add_argument('--pool-size', type=int, default=50000, help='Size of address pool')
    parser.add_argument('--no-warmup', action='store_true', help='Don\'t initialize address pool upfront')
    parser.add_argument('--subset-percentage', type=float, help='Percentage of the dataset to use (optional)')
    args = parser.parse_args()
    
    # Configure address pool
    address_pool.pool_size = args.pool_size
    
    # Run the optimized pipeline
    X_train, X_test, y_train, y_test = process_data_pipeline_optimized(
        args.file_paths, 
        warmup_pool=not args.no_warmup, 
        subset_percentage=args.subset_percentage
    )
    
    # Save results if output path provided
    if args.output:
        base_path = args.output.rstrip('.csv')
        X_train.to_csv(f"{base_path}_X_train.csv", index=False)
        X_test.to_csv(f"{base_path}_X_test.csv", index=False)
        y_train.to_csv(f"{base_path}_y_train.csv", index=False)
        y_test.to_csv(f"{base_path}_y_test.csv", index=False)