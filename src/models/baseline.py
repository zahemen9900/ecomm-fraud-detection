"""
Model training and evaluation utilities for e-commerce fraud detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve,
    average_precision_score, roc_curve, classification_report
)
import joblib
import os


def train_baseline_model(X_train: pd.DataFrame, 
                       y_train: pd.Series,
                       model_type: str = 'logistic') -> Any:
    """
    Train a baseline model for fraud detection.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train ('logistic' or 'random_forest')
        
    Returns:
        Trained model
    """
    if model_type == 'logistic':
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, 
                  X_test: pd.DataFrame, 
                  y_test: pd.Series,
                  threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        threshold: Classification threshold
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'avg_precision': average_precision_score(y_test, y_proba)
    }
    
    return metrics


def print_evaluation_report(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model
    """
    print(f"\n{'-' * 20} {model_name} Evaluation {'-' * 20}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1 Score:     {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"Avg Precision:{metrics['avg_precision']:.4f}")
    print(f"{'-' * 60}")


def plot_confusion_matrix(y_true: pd.Series, 
                         y_pred: pd.Series,
                         title: str = "Confusion Matrix") -> plt.Figure:
    """
    Plot a confusion matrix for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        ax=ax,
        cbar=False
    )
    
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.set_xticklabels(['Not Fraud', 'Fraud'])
    ax.set_yticklabels(['Not Fraud', 'Fraud'])
    
    return fig


def plot_roc_curve(y_true: pd.Series, 
                  y_scores: np.ndarray,
                  title: str = "ROC Curve") -> plt.Figure:
    """
    Plot ROC curve for model evaluation.
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    
    return fig


def plot_precision_recall_curve(y_true: pd.Series, 
                              y_scores: np.ndarray,
                              title: str = "Precision-Recall Curve") -> plt.Figure:
    """
    Plot precision-recall curve for model evaluation.
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, label=f'AP = {avg_precision:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    return fig


def save_model(model: Any, model_path: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        model_path: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}")


def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    return joblib.load(model_path)


def get_feature_importance(model: Any, 
                         feature_names: List[str], 
                         top_n: int = 20) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importances
    """
    if hasattr(model, 'coef_'):
        # For linear models
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models
        importance = model.feature_importances_
    else:
        raise ValueError("Model doesn't support feature importance extraction")
    
    # Create a DataFrame with feature names and importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance and select top N
    feature_importance = feature_importance.sort_values(
        'Importance', ascending=False
    ).head(top_n)
    
    return feature_importance


def plot_feature_importance(feature_importance: pd.DataFrame,
                          title: str = "Feature Importance") -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        feature_importance: DataFrame with feature importance
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importance,
        ax=ax
    )
    
    ax.set_title(title)
    
    return fig