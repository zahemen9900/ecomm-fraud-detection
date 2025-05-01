# Web Application - E-commerce Fraud Detection Demo

This directory contains a simple Flask web application to demonstrate the e-commerce fraud detection model.

## Overview

The web application simulates an e-commerce checkout process. Users can select a product, fill in transaction details, and submit the transaction. The backend then uses the trained fraud detection model to predict whether the transaction is likely fraudulent or not and displays the result.

## Prerequisites

*   Python 3.11 (Will be installed via Conda)
*   Conda (Anaconda or Miniconda distribution)

## Setup

1.  **Navigate to the webapp directory:**
    ```bash
    cd path/to/ecomm-fraud-detection/webapp
    ```

2.  **Create and activate a Conda environment:**
    Replace `webapp-env` with your preferred environment name.
    ```bash
    # Create the environment with Python 3.11
    conda create --name webapp-env python=3.11

    # Activate the environment
    conda activate webapp-env
    ```

3.  **Install dependencies:**
    Make sure you are in the `webapp` directory and the `webapp-env` environment is active.
    ```bash
    # Install dependencies using pip within the Conda environment
    pip install -r requirements.txt
    ```
    *Note: This installs only the dependencies required for the web application itself (like Flask). The core ML model dependencies should be managed in a separate Conda environment based on the root project configuration (e.g., `conda_config.yaml`).*

## Running the Application

1.  **Ensure you are in the `webapp` directory** and the `webapp-env` Conda environment is activated.

2.  **Run the Flask application:**
    ```bash
    python app.py
    ```

3.  **Access the application:**
    Open your web browser and go to the URL provided by Flask (usually `http://127.0.0.1:5000` or `http://localhost:5000`).

## Usage

*   The main page displays several products.
*   Click the "Simulate Transaction" button for a product.
*   The transaction details form will appear, pre-filled with the product's price and category.
*   Fill in the remaining customer and transaction details.
*   Click "Submit Transaction".
*   The application will process the details using the fraud detection model and display the prediction result (Approved or Blocked) along with the fraud probability if applicable.
*   Use the "Back to Products" button on the form to return to the product selection screen. 