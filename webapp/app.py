import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from pathlib import Path
import sys

# --- Add src directory to Python path ---
# Get the absolute path of the current file (app.py)
current_file_path = Path(__file__).resolve()
# Get the parent directory (webapp)
webapp_dir = current_file_path.parent
# Get the project root directory (assuming webapp is directly under the root)
root_dir = webapp_dir.parent
# Construct the path to the src directory within webapp
src_dir = webapp_dir / 'src'
# Add the src directory to sys.path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# --- Import the preprocessing function ---
try:
    # Now import from the src directory within webapp
    from preprocessing import preprocess_single_transaction, model_features # Import model_features list
    print("Successfully imported preprocess_single_transaction and model_features from webapp/src")
except ImportError as e:
    print(f"Error importing preprocessing function or features: {e}")
    # Define dummy function and features if import fails
    def preprocess_single_transaction(data, **kwargs):
        print("WARNING: Using dummy preprocess_single_transaction function.")
        # Return a dummy DataFrame structure based on expected features
        dummy_cols = ['Transaction Amount', 'Quantity', 'Amount_per_Item', 'Same_Address', 'Transaction Hour', 'Is Weekend'] # Example cols
        return pd.DataFrame([[0] * len(dummy_cols)], columns=dummy_cols)
    model_features = None # Set model_features to None if import fails

# --- Configuration ---
app = Flask(__name__)
WEBAPP_DIR = Path(__file__).parent
MODEL_DIR = WEBAPP_DIR / 'models'
PREPROCESSOR_DIR = MODEL_DIR / 'preprocessors' # Define preprocessor dir
MODEL_PATH = MODEL_DIR / 'random_forest.pkl'

# --- Load Model ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None # Set model to None if loading fails
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Load Preprocessors ---
# Moved loading logic to webapp/src/preprocessing.py
# We only need the model_features list here for potential validation if needed,
# but the primary loading happens within the preprocessing script.
print(f"Expecting preprocessors to be loaded by webapp/src/preprocessing.py from {PREPROCESSOR_DIR}")
if model_features is None:
    print("Warning: model_features list was not imported from preprocessing.py. Column alignment might fail.")


# --- Flask Routes ---
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if model_features is None: # Check if features list is loaded
         return jsonify({'error': 'Preprocessing components (feature list) not loaded'}), 500

    try:
        # Get data from the POST request
        data = request.get_json()
        print(f"Received data for prediction: {data}")

        if not data:
            return jsonify({'error': 'No input data received'}), 400

        # --- Preprocess the single transaction using the dedicated function ---
        # The function now handles loading preprocessors internally
        processed_df = preprocess_single_transaction(data)

        if processed_df is None or processed_df.empty:
             # Handle case where preprocessing failed (e.g., preprocessors not found)
             return jsonify({'error': 'Preprocessing failed. Check server logs.'}), 500

        print(f"Processed DataFrame columns: {processed_df.columns.tolist()}\n\n")
        print(f"Processed DataFrame head:\n{processed_df.head()}\n\n")
        print(f"Time of transaction: {data['Transaction Hour']}\n\n")

        # --- Optional: Verify columns match model's expectations ---
        # The preprocess_single_transaction should already handle this, but an extra check:
        if list(processed_df.columns) != model_features:
             print("CRITICAL ERROR: Columns after preprocessing do not match expected model features!")
             print(f"Expected: {model_features}")
             print(f"Got: {list(processed_df.columns)}")
             # You might want to return an error here instead of proceeding
             # Re-align just in case, though this indicates a bug in preprocessing.py
             try:
                 processed_df = processed_df.reindex(columns=model_features, fill_value=0)
                 print("Attempted to reindex DataFrame to match model features.")
             except Exception as reindex_e:
                 print(f"Failed to reindex DataFrame: {reindex_e}")
                 return jsonify({'error': 'Column mismatch after preprocessing and reindex failed.'}), 500


        # --- Make Prediction ---
        prediction = model.predict(processed_df)
        # Ensure predict_proba exists and model is fitted
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(processed_df)[:, 1] # Probability of class 1 (Fraud)
            probability_percentage = f"{probability[0] * 100:.2f}%"
        else:
            probability_percentage = "N/A" # Handle cases where model doesn't support predict_proba

        result_status = "Blocked: Potential Fraud Detected" if prediction[0] == 1 else "Approved"

        print(f"Prediction: {result_status}, Probability: {probability_percentage}")

        # Return the result
        return jsonify({
            'status': result_status,
            'fraud_probability': probability_percentage
        })

    except ValueError as ve:
         print(f"Value Error during prediction: {ve}")
         # Check if it's a missing input field error from preprocessing
         if "Missing input field" in str(ve):
              return jsonify({'error': f'{ve}'}), 400
         # Check if it's an invalid value error from preprocessing
         elif "Invalid input value" in str(ve):
              return jsonify({'error': f'{ve}'}), 400
         else:
              # General value error during model prediction (e.g., shape mismatch)
              return jsonify({'error': f'Invalid data after processing: {ve}'}), 400
    except KeyError as ke:
        # This might occur if preprocess_single_transaction fails to create a required column
        print(f"Key Error during prediction (likely missing feature): {ke}")
        return jsonify({'error': f'Missing expected feature for model: {ke}'}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to console
        return jsonify({'error': f'An internal error occurred: {e}'}), 500

if __name__ == '__main__':
    # Make sure to set debug=False in a production environment
    # Ensure the PREPROCESSOR_DIR exists before starting
    if not PREPROCESSOR_DIR.exists():
        print(f"Warning: Preprocessor directory {PREPROCESSOR_DIR} does not exist. Preprocessing will likely fail.")
        # Optionally create it:
        # PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)
        # print(f"Created directory: {PREPROCESSOR_DIR}")

    app.run(debug=True, host='0.0.0.0', port=5001)