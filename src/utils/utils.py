"""
Utility functions for the fraud detection project.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
import datetime
import json
import time


def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        Project root directory path
    """
    # This assumes the script is executed from within the project
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", ".."))


def ensure_dir_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_data_path() -> str:
    """
    Get the path to the data directory.
    
    Returns:
        Path to the data directory
    """
    return os.path.join(get_project_root(), "data")


def get_models_path() -> str:
    """
    Get the path to the models directory.
    
    Returns:
        Path to the models directory
    """
    models_dir = os.path.join(get_project_root(), "models")
    ensure_dir_exists(models_dir)
    return models_dir


def get_results_path() -> str:
    """
    Get the path to the results directory.
    
    Returns:
        Path to the results directory
    """
    results_dir = os.path.join(get_project_root(), "results")
    ensure_dir_exists(results_dir)
    return results_dir


def generate_timestamp() -> str:
    """
    Generate a timestamp string.
    
    Returns:
        Timestamp string in YYYYMMDD_HHMMSS format
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_dataframe(df: pd.DataFrame, filename: str, directory: Optional[str] = None) -> str:
    """
    Save a dataframe to CSV.
    
    Args:
        df: DataFrame to save
        filename: Filename (without extension)
        directory: Directory to save to (default: results)
        
    Returns:
        Path to saved file
    """
    if directory is None:
        directory = get_results_path()
    
    ensure_dir_exists(directory)
    
    # Add timestamp to filename
    timestamp = generate_timestamp()
    full_filename = f"{filename}_{timestamp}.csv"
    file_path = os.path.join(directory, full_filename)
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")
    
    return file_path


def save_results(results: Dict, filename: str, directory: Optional[str] = None) -> str:
    """
    Save results dictionary to JSON.
    
    Args:
        results: Results dictionary
        filename: Filename (without extension)
        directory: Directory to save to (default: results)
        
    Returns:
        Path to saved file
    """
    if directory is None:
        directory = get_results_path()
    
    ensure_dir_exists(directory)
    
    # Add timestamp to filename
    timestamp = generate_timestamp()
    full_filename = f"{filename}_{timestamp}.json"
    file_path = os.path.join(directory, full_filename)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save to JSON
    with open(file_path, 'w') as f:
        json.dump(results, f, default=convert_for_json, indent=4)
    
    print(f"Results saved to {file_path}")
    
    return file_path


def save_figure(fig: plt.Figure, filename: str, directory: Optional[str] = None) -> str:
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure
        filename: Filename (without extension)
        directory: Directory to save to (default: results)
        
    Returns:
        Path to saved file
    """
    if directory is None:
        directory = os.path.join(get_results_path(), "figures")
    
    ensure_dir_exists(directory)
    
    # Add timestamp to filename
    timestamp = generate_timestamp()
    full_filename = f"{filename}_{timestamp}.png"
    file_path = os.path.join(directory, full_filename)
    
    # Save figure
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {file_path}")
    
    return file_path


def timer(func):
    """
    Decorator to time the execution of a function.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return wrapper