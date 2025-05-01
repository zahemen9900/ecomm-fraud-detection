import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# --- Configuration ---
# Assume preprocessors are saved in webapp/models/preprocessors
# This path needs to match where you save them using the main script
PREPROCESSOR_DIR = Path(__file__).parent.parent / 'models' / 'preprocessors'

# --- Load Preprocessors ---
try:
    print(f"Loading preprocessors from: {PREPROCESSOR_DIR}")
    ordinal_encoder = joblib.load(PREPROCESSOR_DIR / 'ordinal_encoder.joblib')
    ordinal_encoder_cols = joblib.load(PREPROCESSOR_DIR / 'ordinal_encoder_cols.joblib')
    model_features = joblib.load(PREPROCESSOR_DIR / 'model_features.joblib')
    # scaler = joblib.load(PREPROCESSOR_DIR / 'scaler.joblib') # Load if you used scaling
    print("Preprocessors loaded successfully.")
    print(f"Ordinal Encoder Columns: {ordinal_encoder_cols}")
    print(f"Model Features ({len(model_features)}): {model_features}")

except FileNotFoundError as e:
    print(f"Error loading preprocessors: {e}. Ensure they are saved in {PREPROCESSOR_DIR}")
    # Set to None so the app can still run but prediction will fail
    ordinal_encoder = None
    ordinal_encoder_cols = None
    model_features = None
    # scaler = None
except Exception as e:
    print(f"An unexpected error occurred loading preprocessors: {e}")
    ordinal_encoder = None
    ordinal_encoder_cols = None
    model_features = None
    # scaler = None

def preprocess_single_transaction(data_dict: dict) -> pd.DataFrame:
    """
    Preprocesses a single transaction dictionary for prediction.

    Args:
        data_dict: Dictionary containing raw transaction data from the web form.
                   Keys should match the original dataset columns needed for feature engineering.

    Returns:
        A single-row DataFrame ready for model prediction, with columns matching
        the training data. Returns None if preprocessors are not loaded.
    """
    if not all([ordinal_encoder, ordinal_encoder_cols, model_features]):
         print("Error: Preprocessors not loaded. Cannot preprocess transaction.")
         # Returning an empty DataFrame or raising an error might be better
         # depending on how app.py handles it.
         # For now, return an empty DF with expected columns to avoid breaking app.py structure
         # but prediction will likely fail later.
         return pd.DataFrame(columns=model_features if model_features else [])


    print(f"Preprocessing input data: {data_dict}")

    # --- 1. Convert input dict to DataFrame ---
    # Ensure correct data types from the form (Flask usually gives strings)
    try:
        data_dict['Transaction Amount'] = float(data_dict['Transaction Amount'])
        data_dict['Quantity'] = int(data_dict['Quantity'])
        data_dict['Customer Age'] = int(data_dict['Customer Age'])
        data_dict['Account Age Days'] = int(data_dict['Account Age Days'])
        # Transaction Hour might come directly or need calculation
        if 'Transaction Hour' in data_dict:
             data_dict['Transaction Hour'] = int(data_dict['Transaction Hour'])
        else:
             # If not provided, use current hour (adjust timezone if needed)
             data_dict['Transaction Hour'] = datetime.now().hour

        # Handle Transaction Date if needed for feature engineering
        # If the form doesn't provide it, use current date/time
        if 'Transaction Date' not in data_dict:
             data_dict['Transaction Date'] = pd.Timestamp.now()
        else:
             # Attempt to parse if provided (e.g., hidden field or calculated)
             try:
                 data_dict['Transaction Date'] = pd.to_datetime(data_dict['Transaction Date'])
             except ValueError:
                 print("Warning: Could not parse Transaction Date, using current time.")
                 data_dict['Transaction Date'] = pd.Timestamp.now()

    except KeyError as e:
        print(f"Error: Missing expected key in input data: {e}")
        raise ValueError(f"Missing input field: {e}") from e
    except ValueError as e:
        print(f"Error: Invalid data type for a field: {e}")
        raise ValueError(f"Invalid input value: {e}") from e


    df = pd.DataFrame([data_dict])


    # --- 2. Feature Engineering (matching training script, excluding address transformation) ---
    print("Engineering features...")
    # Amount per Item
    # Avoid division by zero if Quantity could be 0
    df['Amount_per_Item'] = df['Transaction Amount'] / df['Quantity'].replace(0, 1)

    # Same Address (assuming Shipping and Billing addresses are provided in data_dict)
    if 'Shipping Address' in df.columns and 'Billing Address' in df.columns:
        df['Same_Address'] = (df['Shipping Address'] == df['Billing Address']).astype(int)
    else:
        # If addresses aren't provided/needed for the model, set a default (e.g., 1 or 0)
        # Or raise an error if they ARE needed by the model features list
        print("Warning: Shipping/Billing Address not found in input. Setting Same_Address=1 (default).")
        df['Same_Address'] = 1 # Default or handle as needed

    # Time-based features (using the Transaction Date)
    transaction_date = df['Transaction Date'].iloc[0] # Get the single timestamp
    df['Transaction Day'] = transaction_date.day
    df['Transaction Month'] = transaction_date.month
    df['Transaction Year'] = transaction_date.year
    df['Transaction DayOfWeek'] = transaction_date.dayofweek
    df['Is Weekend'] = df['Transaction DayOfWeek'].isin([5, 6]).astype(int)

    # Transaction Recency - This is tricky for real-time.
    # Option 1: Calculate relative to 'now'. Might differ from training.
    # df['Transaction_Recency_Days'] = 0 # Assume 0 days for a live transaction
    # Option 2: Calculate relative to a fixed date used in training (if available)
    # fixed_max_date = pd.to_datetime('YYYY-MM-DD') # Replace with actual max date from training
    # df['Transaction_Recency_Days'] = (fixed_max_date - transaction_date).days
    # Option 3: If not highly important, maybe exclude or set to a mean/median.
    # For now, let's set to 0, assuming it's a live transaction.
    df['Transaction_Recency_Days'] = 0
    print(f"Engineered features: {df.columns.tolist()}")


    # --- 3. Encoding ---
    print("Applying encoding...")
    # Ordinal Encoding (using loaded encoder)
    if 'Customer Location' in df.columns and 'Customer Location' in ordinal_encoder_cols:
        # Handle potential new/unknown locations seen in production
        # The encoder was saved with handle_unknown='use_encoded_value', unknown_value=-1
        loc_col = df[['Customer Location']].fillna('Unknown') # Fill NaNs before encoding
        df['Customer Location'] = ordinal_encoder.transform(loc_col)
        print("Applied OrdinalEncoder to Customer Location.")
    elif 'Customer Location' in model_features:
         print("Warning: 'Customer Location' expected by model but not found or not in ordinal cols.")
         # Decide how to handle: error, or fill with default (e.g., -1 for unknown)
         df['Customer Location'] = -1


    # One-Hot Encoding (for Payment Method, Product Category, Device Used)
    categorical_cols_ohe = ['Payment Method', 'Product Category', 'Device Used']
    # Filter to cols actually present in the input df
    cols_to_encode = [col for col in categorical_cols_ohe if col in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True, dummy_na=False)
        print(f"Applied OneHotEncoder to: {cols_to_encode}")


    # --- 4. Scaling (Optional - Apply if used during training) ---
    # print("Applying scaling...")
    # numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # if scaler and numeric_cols:
    #     df[numeric_cols] = scaler.transform(df[numeric_cols])
    #     print("Applied Scaler.")


    # --- 5. Ensure Final Columns Match Model Features ---
    print("Aligning columns with model features...")
    # Add missing columns (expected by model but not in current df) and fill with 0
    missing_cols = set(model_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
        # Important: Ensure the dtype matches the model's expectation (usually float for numeric/encoded)
        # If the missing column is boolean (from OHE), set to False or 0
        if col.startswith(tuple(f"{c}_" for c in categorical_cols_ohe)) or col in ['Same_Address', 'Is Weekend']:
             df[col] = df[col].astype(bool) # Or int(0)
        else:
             df[col] = df[col].astype(float) # Default to float for numeric/ordinal


    # Select and reorder columns to exactly match the order expected by the model
    df = df[model_features]
    print(f"Final columns ({len(df.columns)}): {df.columns.tolist()}")

    # --- 6. Final Check ---
    # Check for NaNs introduced during processing
    if df.isnull().values.any():
        print("Warning: NaNs detected after preprocessing. Filling with 0.")
        print(df.isnull().sum())
        df = df.fillna(0) # Or use a more sophisticated imputation if needed

    return df
