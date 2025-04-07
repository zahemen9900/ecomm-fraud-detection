"""
Data loading and preprocessing module for e-commerce fraud detection.
Handles loading CSV files, cleaning, and feature engineering.
"""

import os
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


# Try to import tqdm, use a simple pass-through if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load transaction data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing transaction data
    """
    return pd.read_csv(file_path)


def _parallel_load_data(file_paths: List[str], n_jobs=None) -> pd.DataFrame:
    """
    Load multiple CSV files in parallel and merge them.
    
    Args:
        file_paths: List of paths to CSV files
        n_jobs: Number of parallel jobs to run (defaults to None, which uses all available cores)
        
    Returns:
        Merged DataFrame
    """
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Loading {len(file_paths)} files using {n_jobs} workers...")
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all file loading jobs to the executor
        future_to_file = {executor.submit(load_data, file_path): file_path for file_path in file_paths}
        
        # Create a placeholder for results
        dataframes = []
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_file), total=len(file_paths), desc="Loading files"):
            file_path = future_to_file[future]
            try:
                df = future.result()
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Concatenate all dataframes
    return pd.concat(dataframes, ignore_index=True)


def merge_datasets(file_paths: List[str]) -> pd.DataFrame:
    """
    Merge multiple transaction datasets.
    
    Args:
        file_paths: List of paths to CSV files
        
    Returns:
        Merged DataFrame
    """
    # Use parallel loading for multiple files
    if len(file_paths) > 1:
        return _parallel_load_data(file_paths)
    else:
        # For a single file, use the standard loader
        return load_data(file_paths[0])


def generate_ghanaian_location():
    """
    Generate a random Ghanaian location name (city/town with neighborhood or qualifier).
    
    Returns:
        Tuple of (location_string, region)
    """
    
    # Ghanaian regions
    regions = [
        'Greater Accra', 'Ashanti', 'Central', 'Western', 'Eastern',
        'Volta', 'Northern', 'Upper East', 'Upper West', 'Bono',
        'Ahafo', 'Bono East', 'North East', 'Oti', 'Savannah',
        'Western North'
    ]
    
    # Major cities/towns by region
    cities = {
        'Greater Accra': ['Accra', 'Tema', 'Madina', 'Teshie', 'Nungua', 'La', 'Osu', 'Adenta', 'Ashaiman', 'Dome', 'Amasaman', 'Ga', 'Dansoman', 'Achimota'],
        'Ashanti': ['Kumasi', 'Obuasi', 'Ejisu', 'Bekwai', 'Konongo', 'Mampong', 'Offinso', 'Juaben', 'Asokore Mampong', 'Effiduase', 'Asante Akim', 'Jacobu'],
        'Central': ['Cape Coast', 'Winneba', 'Kasoa', 'Mankessim', 'Elmina', 'Swedru', 'Apam', 'Saltpond', 'Assin Fosu', 'Dunkwa-on-Offin', 'Komenda', 'Moree'],
        'Western': ['Takoradi', 'Sekondi', 'Tarkwa', 'Axim', 'Nkroful', 'Prestea', 'Bogoso', 'Half Assini', 'Elubo', 'Shama', 'Daboase'],
        'Eastern': ['Koforidua', 'Nkawkaw', 'Mpraeso', 'Akropong', 'Nsawam', 'Suhum', 'Somanya', 'Kibi', 'Asamankese', 'Akosombo', 'Aburi', 'Oda'],
        'Volta': ['Ho', 'Keta', 'Hohoe', 'Kpando', 'Anloga', 'Aflao', 'Kpeve', 'Dzodze', 'Akatsi', 'Sogakope', 'Abor', 'Adidome'],
        'Northern': ['Tamale', 'Yendi', 'Savelugu', 'Bimbilla', 'Gushegu', 'Karaga', 'Kpandai', 'Zabzugu', 'Wulensi', 'Kumbungu'],
        'Upper East': ['Bolgatanga', 'Navrongo', 'Bawku', 'Zebilla', 'Paga', 'Binaba', 'Garu', 'Sandema', 'Pusiga', 'Tongo'],
        'Upper West': ['Wa', 'Lawra', 'Tumu', 'Nandom', 'Jirapa', 'Lambussie', 'Gwollu', 'Hamile', 'Wechiau', 'Funsi'],
        'Bono': ['Sunyani', 'Berekum', 'Dormaa Ahenkro', 'Wenchi', 'Nsoatre', 'Sampa', 'New Drobo', 'Odumase', 'Nsawkaw'],
        'Ahafo': ['Goaso', 'Kenyasi', 'Hwidiem', 'Kukuom', 'Bechem', 'Duayaw Nkwanta', 'Mim', 'Sankore'],
        'Bono East': ['Techiman', 'Kintampo', 'Nkoranza', 'Atebubu', 'Yeji', 'Kwame Danso', 'Jema', 'Prang', 'Busunya'],
        'North East': ['Nalerigu', 'Walewale', 'Gambaga', 'Chereponi', 'Bunkpurugu', 'Yunyoo', 'Langbinsi'],
        'Oti': ['Dambai', 'Jasikan', 'Kadjebi', 'Nkwanta', 'Kete Krachi', 'Worawora', 'Likpe', 'Nkonya', 'Brewaniase'],
        'Savannah': ['Damongo', 'Salaga', 'Bole', 'Buipe', 'Sawla', 'Daboya', 'Bamboi', 'Mpaha', 'Larabanga'],
        'Western North': ['Sefwi Wiawso', 'Bibiani', 'Enchi', 'Juaboso', 'Dadieso', 'Akontombra', 'Bodi', 'Essam', 'Adabokrom']
    }
    
    # Neighborhood/area descriptors for added variety
    neighborhood_descriptors = [
        'Central', 'Old Town', 'New Town', 'East', 'West', 'North', 'South',
        'Ridge', 'Zongo', 'Extension', 'Estates', 'Market Area', 'Township',
        'Residential Area', 'Business District', 'Industrial Area', 'Cantonments',
        'Suburb', 'Downtown', 'Uptown', 'Harbor Area', 'Lakeside', 'University Area',
        'Commercial Center', 'Shopping District', 'Old Quarter', 'New Development',
        'Airport Area', 'Beach Road', 'High Street', 'Low Cost', 'Middle Income'
    ]
    
    # Location generation patterns
    location_pattern = random.randint(1, 5)
    
    # Randomly select a region
    region = random.choice(regions)
    
    # Generate location based on pattern
    if location_pattern == 1:
        # Simple city name
        city = random.choice(cities[region])
        location = city
    elif location_pattern == 2:
        # City with neighborhood
        city = random.choice(cities[region])
        neighborhood = random.choice(neighborhood_descriptors)
        location = f"{city} {neighborhood}"
    elif location_pattern == 3:
        # Directional prefix + city
        city = random.choice(cities[region])
        prefix = random.choice(['New', 'East', 'West', 'North', 'South', 'Upper', 'Lower', 'Central'])
        location = f"{prefix} {city}"
    elif location_pattern == 4:
        # City with region qualifier
        city = random.choice(cities[region])
        location = f"{city}, {region}"
    else:
        # Random street name in city
        city = random.choice(cities[region])
        streets = [
            'High Street', 'Market Road', 'Main Street', 'Harbour Area', 
            'Station Road', 'Beach Road', 'Airport Road', 'University Road',
            'Castle Road', 'Downtown', 'Commercial District'
        ]
        street = random.choice(streets)
        location = f"{street}, {city}"
    
    # 20% chance to add a location type specifier
    if random.random() < 0.2:
        specifers = ['District', 'Area', 'Suburb', 'Neighborhood', 'Quarter', 'Zone', 'Town', 'Village']
        location += f" {random.choice(specifers)}"
        
    return location, region


def generate_ghanaian_address():
    """Generate a random Ghanaian address."""
    
    # Ghanaian regions
    regions = [
        'Greater Accra', 'Ashanti', 'Central', 'Western', 'Eastern',
        'Volta', 'Northern', 'Upper East', 'Upper West', 'Bono',
        'Ahafo', 'Bono East', 'North East', 'Oti', 'Savannah',
        'Western North'
    ]
    
    # Get a location and region
    location, region = generate_ghanaian_location()
    city = location.split(',')[0] if ',' in location else location.split(' ')[0]
    
    # Common street types
    street_types = ['Street', 'Road', 'Avenue', 'Close', 'Link', 'Drive', 
                   'Lane', 'Crescent', 'Highway', 'Circuit', 'Way', 'Boulevard', 
                   'Bypass', 'Path', 'Square', 'Alley', 'Junction', 'Extension', 
                   'Terrace', 'Loop', 'Place']
    
    # Common Ghanaian street naming patterns
    street_patterns = [
        'Market', 'Hospital', 'Stadium', 'Ring', 'High', 'Beach', 'Castle',
        'Independence', 'Liberation', 'Republic', 'Commercial', 'Industrial',
        'Harbour', 'Airport', 'Palace', 'University', 'School', 'Mall',
        'Cantonments', 'Kojo Thompson', 'Oxford', 'Kwame Nkrumah', 'Atta Mills',
        'John Mahama', 'Akufo-Addo', 'Legon', 'Osu', 'Ridge', 'Spintex', 'Graphic',
        'Tetteh Quarshie', 'Gold', 'Silver', 'Diamond', 'Royal', 'Unity', 'Peace',
        'Freedom', 'Justice', 'Indece', 'Parliament', 'Cocoa', 'Volta', 'Lake',
        'Ghana', 'Africa', 'Adenta', 'Kasoa', 'Tema', 'Labadi', 'Aburi', 'Makola',
        '28th February', '6th March', 'Legon Botanical Gardens', 'Accra Central'
    ]
    
    # House numbering patterns (GPS-style and traditional)
    house_number_patterns = [
        f"GA-{random.randint(100, 999)}-{random.randint(1000, 9999)}",  # GPS style
        f"H/No. {random.randint(1, 999)}", # Traditional style
        f"Plot {random.randint(1, 500)}",  # Plot style
        f"Digital Address: {random.choice('ABCDEFGH')}{random.randint(1,9)}-{random.randint(100,999)}-{random.randint(1000,9999)}"  # Digital address
    ]
    
    # Landmarks (optional)
    landmarks = [
        'Near Central Market', 'Behind Post Office', 'Opposite Police Station',
        'Near Church', 'Behind Mosque', 'Next to School', 'Near Hospital',
        'Opposite Bank', 'Near Chief Palace', 'Behind Lorry Station'
    ]
    
    # Generate address components
    street_type = random.choice(street_types)
    street_pattern = random.choice(street_patterns)
    house_number = random.choice(house_number_patterns)
    landmark = random.choice(landmarks) if random.random() < 0.3 else ""  # 30% chance to include landmark
    
    # Construct address in different formats
    address_formats = [
        f"{house_number}, {street_pattern} {street_type}\n{location}, {region} Region",
        f"{house_number}\n{street_pattern} {street_type}, {landmark}\n{location}, {region} Region",
        f"{house_number}, {street_pattern} {street_type}\n{landmark}\n{location}, {region}",
        f"{street_pattern} {street_type}\n{house_number}, {location}\n{region} Region"
    ]
    
    return random.choice(address_formats)


def _process_address_batch(batch):
    """
    Helper function to process a batch of address transformations in parallel.
    
    Args:
        batch: Tuple of (start_idx, end_idx, df_batch, addresses_to_transform)
        
    Returns:
        Tuple of (start_idx, end_idx, transformed_shipping, transformed_billing, transformed_location)
    """
    start_idx, end_idx, df_batch, _ = batch
    batch_size = end_idx - start_idx
    
    # Generate addresses and locations
    ghanaian_locations = [generate_ghanaian_location()[0] for _ in range(batch_size)]
    ghanaian_addresses = [generate_ghanaian_address() for _ in range(batch_size)]
    
    transformed_shipping = []
    transformed_billing = []
    
    # Process addresses in the batch
    for i in range(batch_size):
        local_idx = i + start_idx
        
        # Check if Same_Address exists or needs to be inferred
        if 'Same_Address' in df_batch.columns:
            same_address = df_batch.iloc[i]['Same_Address'] == 1
        else:
            # Infer from matching addresses
            same_address = df_batch.iloc[i]['Shipping Address'] == df_batch.iloc[i]['Billing Address']
            
        if same_address:
            # Use same address for both shipping and billing
            transformed_shipping.append(ghanaian_addresses[i])
            transformed_billing.append(ghanaian_addresses[i])
        else:
            # Generate different addresses for shipping and billing
            transformed_shipping.append(ghanaian_addresses[i])
            transformed_billing.append(generate_ghanaian_address())
    
    return start_idx, end_idx, transformed_shipping, transformed_billing, ghanaian_locations


def transform_to_ghanaian_addresses(df):
    """
    Transform addresses in the dataset to Ghanaian addresses.
    
    Args:
        df: DataFrame containing 'Shipping Address', 'Billing Address', and 'Customer Location' columns
        
    Returns:
        DataFrame with transformed addresses
    """
    # Create copy to avoid modifying original
    df_transformed = df.copy()
    n_transactions = len(df)
    
    start_time = time.time()
    print(f"Transforming {n_transactions} addresses to Ghanaian format...")
    
    # Determine if parallelization is beneficial (for small datasets, parallel overhead is too high)
    if n_transactions < 5000:
        # For small datasets, use the simple approach
        ghanaian_locations = [generate_ghanaian_location()[0] for _ in range(n_transactions)]
        ghanaian_addresses = [generate_ghanaian_address() for _ in range(n_transactions)]
        
        for i in range(n_transactions):
            # Transform Customer Location
            df_transformed.loc[i, 'Customer Location'] = ghanaian_locations[i]
            
            if 'Same_Address' in df_transformed.columns:
                same_address = df_transformed.loc[i, 'Same_Address'] == 1
            else:
                # Infer from matching addresses
                same_address = df_transformed.loc[i, 'Shipping Address'] == df_transformed.loc[i, 'Billing Address']
                
            if same_address:
                # Use same address for both shipping and billing
                df_transformed.loc[i, 'Shipping Address'] = ghanaian_addresses[i]
                df_transformed.loc[i, 'Billing Address'] = ghanaian_addresses[i]
            else:
                # Generate different addresses for shipping and billing
                df_transformed.loc[i, 'Shipping Address'] = ghanaian_addresses[i]
                df_transformed.loc[i, 'Billing Address'] = generate_ghanaian_address()
    else:
        # For larger datasets, use parallel processing
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        batch_size = min(1000, max(100, n_transactions // (n_jobs * 2)))  # Adjust batch size based on dataset size
        
        # Split into batches
        batches = []
        for i in range(0, n_transactions, batch_size):
            end_idx = min(i + batch_size, n_transactions)
            batches.append((i, end_idx, df.iloc[i:end_idx], None))
        
        print(f"Processing {len(batches)} batches in parallel using {n_jobs} workers...")
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_process_address_batch, batch) for batch in batches]
            
            # Use tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing address batches"):
                start_idx, end_idx, transformed_shipping, transformed_billing, transformed_locations = future.result()
                
                # Update dataframe with transformed addresses
                for i, (idx, ship, bill, loc) in enumerate(zip(
                    range(start_idx, end_idx),
                    transformed_shipping,
                    transformed_billing,
                    transformed_locations
                )):
                    df_transformed.loc[idx, 'Shipping Address'] = ship
                    df_transformed.loc[idx, 'Billing Address'] = bill
                    df_transformed.loc[idx, 'Customer Location'] = loc
    
    print(f"Address transformation completed in {time.time() - start_time:.2f} seconds")
    return df_transformed


def _process_time_features(df_chunk):
    """Helper function to process time-based features for a chunk of data in parallel"""
    result = df_chunk.copy()
    
    # Extract time-based features if not already present
    if 'Transaction Hour' not in result.columns:
        result['Transaction Hour'] = result['Transaction Date'].dt.hour
    
    # Extract additional time features
    result['Transaction Day'] = result['Transaction Date'].dt.day
    result['Transaction Month'] = result['Transaction Date'].dt.month
    result['Transaction Year'] = result['Transaction Date'].dt.year
    result['Transaction DayOfWeek'] = result['Transaction Date'].dt.dayofweek
    result['Is Weekend'] = result['Transaction DayOfWeek'].isin([5, 6]).astype(int)
    
    # Transaction recency (days since most recent transaction date)
    max_date = result['Transaction Date'].max()
    result['Transaction_Recency_Days'] = (max_date - result['Transaction Date']).dt.days
    
    return result


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the transaction data by handling missing values,
    removing duplicates, and fixing data types.
    
    Args:
        df: DataFrame with raw transaction data
        
    Returns:
        Cleaned DataFrame
    """
    start_time = time.time()
    print(f"Cleaning dataset with {len(df)} records...")
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Enable pandas parallel operations where supported
    try:
        pd.options.mode.parallel = True
    except:
        pass
    
    # Remove duplicate transactions
    cleaned_df = cleaned_df.drop_duplicates(subset=['Transaction ID'])
    
    # Convert transaction date to datetime
    cleaned_df['Transaction Date'] = pd.to_datetime(cleaned_df['Transaction Date'], errors='coerce')
    
    # Handle missing values
    cleaned_df['Transaction Amount'] = cleaned_df['Transaction Amount'].fillna(cleaned_df['Transaction Amount'].median())
    cleaned_df['Quantity'] = cleaned_df['Quantity'].fillna(1).astype(int)
    
    # Fill categorical missing values with 'unknown'
    categorical_cols = ['Payment Method', 'Product Category', 'Device Used']
    for col in categorical_cols:
        cleaned_df[col] = cleaned_df[col].fillna('unknown')
    
    print(f"Data cleaning completed in {time.time() - start_time:.2f} seconds")
    return cleaned_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on transaction data.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with additional engineered features
    """
    start_time = time.time()
    print(f"Engineering features for {len(df)} records...")
    
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
    
    # Transform addresses to Ghanaian format
    print("Transforming addresses...")
    enhanced_df = transform_to_ghanaian_addresses(enhanced_df)
    
    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
    return enhanced_df


def _one_hot_encode_chunk(chunk, categorical_cols):
    """Helper function to one-hot encode a chunk of data"""
    return pd.get_dummies(chunk, columns=categorical_cols, drop_first=True)


def prepare_data_for_modeling(df: pd.DataFrame, 
                             target_col: str = 'Is Fraudulent',
                             drop_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training by separating features and target.
    
    Args:
        df: DataFrame with all features
        target_col: Name of the target column
        drop_cols: Columns to drop from features
        
    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the target Series
    """
    start_time = time.time()
    print(f"Preparing data for modeling ({len(df)} records)...")
    
    if drop_cols is None:
        drop_cols = ['Transaction ID', 'Customer ID', 'Transaction Date', 
                     'IP Address', 'Shipping Address', 'Billing Address']
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Handle NaN values in target column
    if df_copy[target_col].isna().any():
        nan_count = df_copy[target_col].isna().sum()
        print(f"Warning: Found {nan_count} NaN values in the target column '{target_col}'")
        
        # Option 1: Remove rows with NaN target values
        df_copy = df_copy.dropna(subset=[target_col])
        print(f"Dropped {nan_count} rows with NaN target values. {len(df_copy)} rows remaining.")
        
        # Alternative options (commented out, uncomment if needed):
        # Option 2: Fill NaN targets with most common class
        # most_common = df_copy[target_col].mode()[0]
        # df_copy[target_col] = df_copy[target_col].fillna(most_common)
        # print(f"Filled {nan_count} NaN target values with most common class: {most_common}")
        
        # Option 3: Fill NaN targets with 0 (non-fraudulent)
        # df_copy[target_col] = df_copy[target_col].fillna(0)
        # print(f"Filled {nan_count} NaN target values with 0 (assuming non-fraudulent)")
    
    # Separate target
    y = df_copy[target_col]
    
    # Drop unnecessary columns
    drop_cols = drop_cols + [target_col] if target_col not in drop_cols else drop_cols
    X = df_copy.drop(columns=drop_cols)
    
    # Identify categorical columns for encoding
    categorical_cols = ['Payment Method', 'Product Category', 'Device Used']
    other_categorical_cols = [col for col in X.select_dtypes(include=['object']).columns.tolist() if col not in categorical_cols]
    
    # Apply ordinal encoding to other categorical columns
    if other_categorical_cols:
        print(f"Applying ordinal encoding to {len(other_categorical_cols)} columns...")
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_ordinal = X[other_categorical_cols].copy()
        X_ordinal_encoded = ordinal_encoder.fit_transform(X_ordinal)
        
        # Replace the original columns with encoded versions
        for i, col in enumerate(other_categorical_cols):
            X[col] = X_ordinal_encoded[:, i]
    
    # For larger datasets, use parallel one-hot encoding
    if len(X) > 10000:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        chunk_size = min(5000, max(1000, len(X) // (n_jobs * 2)))
        
        # Split X into chunks
        chunks = [X.iloc[i:i+chunk_size].copy() for i in range(0, len(X), chunk_size)]
        print(f"One-hot encoding in {len(chunks)} parallel chunks using {n_jobs} workers...")
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Create partial function for applying to each chunk
            encode_func = partial(_one_hot_encode_chunk, categorical_cols=categorical_cols)
            
            # Map the function to all chunks
            encoded_chunks = list(tqdm(
                executor.map(encode_func, chunks),
                total=len(chunks),
                desc="One-hot encoding"
            ))
        
        # To handle differences in columns between chunks, get all possible columns
        all_columns = set()
        for chunk in encoded_chunks:
            all_columns.update(chunk.columns)
        
        # Ensure all chunks have the same columns
        for i, chunk in enumerate(encoded_chunks):
            # Add missing columns with zeros
            missing_cols = all_columns.difference(chunk.columns)
            for col in missing_cols:
                encoded_chunks[i][col] = 0
                
        # Combine chunks and ensure we have all columns in the same order
        X = pd.concat(encoded_chunks)
        X = X.fillna(0) # Fill any NaN values that might have been introduced
        X = X.sort_index() # Ensure original order is maintained
    else:
        # For smaller datasets, use standard encoding
        print("Performing one-hot encoding...")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"Data preparation completed in {time.time() - start_time:.2f} seconds")
    return X, y


def process_data_pipeline(file_paths: List[str], subset_percentage=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Full data processing pipeline from loading to train-test split.
    
    Args:
        file_paths: List of paths to CSV files
        subset_percentage: If provided, use only this percentage of the dataset (maintaining class balance)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    total_start_time = time.time()
    print(f"\nStarting data processing pipeline...")
    
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
    
    # Engineer features
    print("\nSTEP 3: Engineering features...")
    df_engineered = engineer_features(df_cleaned)
    
    # Prepare for modeling
    print("\nSTEP 4: Preparing data for modeling...")
    X, y = prepare_data_for_modeling(df_engineered)
    print(f"Final processed dataset has {X.shape[1]} features after encoding")
    
    # Split into training and testing sets
    print("\nSTEP 5: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nEntire data pipeline completed in {time.time() - total_start_time:.2f} seconds")
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage as a script
    import argparse
    
    parser = argparse.ArgumentParser(description='Process e-commerce data for fraud detection modeling')
    parser.add_argument('file_paths', nargs='+', help='Paths to CSV files to process')
    parser.add_argument('--output', type=str, help='Path to save processed data (optional)')
    parser.add_argument('--subset_percentage', type=float, help='Percentage of data to use (optional)')
    args = parser.parse_args()
    
    # Run the pipeline
    X_train, X_test, y_train, y_test = process_data_pipeline(args.file_paths, subset_percentage=args.subset_percentage)
    
    # Save results if output path provided
    if args.output:
        base_path = args.output.rstrip('.csv')
        X_train.to_csv(f"{base_path}_X_train.csv", index=False)
        X_test.to_csv(f"{base_path}_X_test.csv", index=False)
        y_train.to_csv(f"{base_path}_y_train.csv", index=False)
        y_test.to_csv(f"{base_path}_y_test.csv", index=False)
        print(f"Processed data saved to {base_path}_*.csv files")