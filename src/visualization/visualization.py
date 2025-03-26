"""
Visualization utilities for e-commerce fraud detection project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any


def set_visualization_style() -> None:
    """
    Set up matplotlib and seaborn visualization style.
    """
    # Use ggplot style for matplotlib
    plt.style.use('ggplot')
    
    # Set seaborn aesthetics
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Set default figure size
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = 'sans-serif'


def plot_fraud_distribution(df: pd.DataFrame,
                           target_col: str = 'Is Fraudulent',
                           title: str = 'Distribution of Fraudulent Transactions') -> plt.Figure:
    """
    Plot the distribution of fraudulent vs. non-fraudulent transactions.
    
    Args:
        df: DataFrame containing transaction data
        target_col: Name of the target column
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fraud_counts = df[target_col].value_counts()
    fraud_percentage = df[target_col].mean() * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(
        ['Not Fraud', 'Fraud'],
        [fraud_counts[0], fraud_counts[1]],
        color=colors
    )
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 5,
            f'{height:,}',
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    # Add percentage of fraud to title
    title_with_pct = f"{title}\nFraud: {fraud_percentage:.2f}% of transactions"
    ax.set_title(title_with_pct, fontsize=14)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_ylim(0, fraud_counts[0] * 1.1)  # Add some padding for the labels
    
    # Add text box with statistics
    stats_text = (f"Total Transactions: {len(df):,}\n"
                  f"Non-Fraud: {fraud_counts[0]:,} ({100 - fraud_percentage:.2f}%)\n"
                  f"Fraud: {fraud_counts[1]:,} ({fraud_percentage:.2f}%)")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)
    
    return fig


def plot_numeric_distribution(df: pd.DataFrame, 
                            feature: str,
                            target_col: str = 'Is Fraudulent',
                            bins: int = 50,
                            title: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of a numeric feature by fraud status.
    
    Args:
        df: DataFrame containing transaction data
        feature: Name of the numeric feature
        target_col: Name of the target column
        bins: Number of bins for histogram
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    if title is None:
        title = f'Distribution of {feature} by Fraud Status'
    
    # Plot histogram with KDE for non-fraudulent transactions
    sns.histplot(
        df[df[target_col] == 0][feature],
        bins=bins,
        kde=True,
        color='blue',
        alpha=0.6,
        ax=ax1,
        stat='probability',
        label='Not Fraud'
    )
    ax1.set_title(f'{feature} Distribution - Not Fraud', fontsize=12)
    ax1.legend()
    
    # Plot histogram with KDE for fraudulent transactions
    sns.histplot(
        df[df[target_col] == 1][feature],
        bins=bins,
        kde=True,
        color='red',
        alpha=0.6,
        ax=ax2,
        stat='probability',
        label='Fraud'
    )
    ax2.set_title(f'{feature} Distribution - Fraud', fontsize=12)
    ax2.legend()
    
    # Add descriptive statistics
    for ax, fraud_val, label in zip([ax1, ax2], [0, 1], ['Not Fraud', 'Fraud']):
        subset = df[df[target_col] == fraud_val][feature]
        stats_text = (f"Mean: {subset.mean():.2f}\n"
                    f"Median: {subset.median():.2f}\n"
                    f"Std: {subset.std():.2f}\n"
                    f"Min: {subset.min():.2f}\n"
                    f"Max: {subset.max():.2f}")
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
    
    plt.xlabel(feature, fontsize=12)
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    
    return fig


def plot_categorical_distribution(df: pd.DataFrame, 
                                feature: str,
                                target_col: str = 'Is Fraudulent',
                                top_n: int = 10,
                                title: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of a categorical feature by fraud status.
    
    Args:
        df: DataFrame containing transaction data
        feature: Name of the categorical feature
        target_col: Name of the target column
        top_n: Number of top categories to show
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    # Get top N categories by frequency
    top_categories = df[feature].value_counts().head(top_n).index.tolist()
    
    # Filter dataframe to include only top categories
    df_plot = df[df[feature].isin(top_categories)].copy()
    
    # Calculate fraud rate for each category
    fraud_rate = df_plot.groupby(feature)[target_col].mean().sort_values(ascending=False)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    if title is None:
        title = f'Distribution of {feature} by Fraud Status'
    
    # Plot count of transactions by category and fraud status
    sns.countplot(
        x=feature,
        hue=target_col,
        data=df_plot,
        palette=['blue', 'red'],
        ax=ax1,
        order=fraud_rate.index
    )
    ax1.set_title(f'Count of {feature} by Fraud Status', fontsize=12)
    ax1.set_xlabel('')
    ax1.legend(['Not Fraud', 'Fraud'])
    
    # Rotate x-tick labels if they are too long
    if df_plot[feature].astype(str).map(len).max() > 10:
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot fraud rate by category
    sns.barplot(
        x=fraud_rate.index,
        y=fraud_rate.values,
        ax=ax2,
        color='purple'
    )
    ax2.set_title(f'Fraud Rate by {feature}', fontsize=12)
    ax2.set_xlabel(feature)
    ax2.set_ylabel('Fraud Rate')
    
    # Add value labels on the fraud rate bars
    for i, v in enumerate(fraud_rate.values):
        ax2.text(i, v + 0.01, f'{v:.2%}', ha='center', fontsize=10)
    
    # Rotate x-tick labels if they are too long
    if df_plot[feature].astype(str).map(len).max() > 10:
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    
    return fig


def plot_time_patterns(df: pd.DataFrame,
                     date_col: str = 'Transaction Date',
                     target_col: str = 'Is Fraudulent',
                     title: str = 'Temporal Patterns of Fraud') -> plt.Figure:
    """
    Plot fraud rate by hour, day of week, and month.
    
    Args:
        df: DataFrame containing transaction data
        date_col: Name of the date column
        target_col: Name of the target column
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    # Ensure date column is datetime type
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract time components
    df['hour'] = df[date_col].dt.hour
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    
    # Days of week names
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Month names
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot 1: Fraud rate by hour
    hour_fraud = df.groupby('hour')[target_col].mean()
    hour_volume = df.groupby('hour').size() / len(df)
    
    ax1 = axes[0]
    color = 'tab:blue'
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Fraud Rate', color=color)
    ax1.plot(hour_fraud.index, hour_fraud.values, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Fraud Rate by Hour of Day', fontsize=12)
    ax1.set_xticks(range(0, 24, 2))
    
    # Add secondary axis for transaction volume
    ax1_2 = ax1.twinx()
    color = 'tab:red'
    ax1_2.set_ylabel('Transaction Volume (% of total)', color=color)
    ax1_2.plot(hour_volume.index, hour_volume.values, 'o--', color=color, alpha=0.6)
    ax1_2.tick_params(axis='y', labelcolor=color)
    
    # Plot 2: Fraud rate by day of week
    day_fraud = df.groupby('day_of_week')[target_col].mean()
    day_volume = df.groupby('day_of_week').size() / len(df)
    
    ax2 = axes[1]
    color = 'tab:blue'
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Fraud Rate', color=color)
    ax2.plot(day_fraud.index, day_fraud.values, 'o-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_title('Fraud Rate by Day of Week', fontsize=12)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(days)
    
    # Add secondary axis for transaction volume
    ax2_2 = ax2.twinx()
    color = 'tab:red'
    ax2_2.set_ylabel('Transaction Volume (% of total)', color=color)
    ax2_2.plot(day_volume.index, day_volume.values, 'o--', color=color, alpha=0.6)
    ax2_2.tick_params(axis='y', labelcolor=color)
    
    # Plot 3: Fraud rate by month
    month_fraud = df.groupby('month')[target_col].mean()
    month_volume = df.groupby('month').size() / len(df)
    
    ax3 = axes[2]
    color = 'tab:blue'
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Fraud Rate', color=color)
    ax3.plot(month_fraud.index, month_fraud.values, 'o-', color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_title('Fraud Rate by Month', fontsize=12)
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(months)
    
    # Add secondary axis for transaction volume
    ax3_2 = ax3.twinx()
    color = 'tab:red'
    ax3_2.set_ylabel('Transaction Volume (% of total)', color=color)
    ax3_2.plot(month_volume.index, month_volume.values, 'o--', color=color, alpha=0.6)
    ax3_2.tick_params(axis='y', labelcolor=color)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_correlation_heatmap(df: pd.DataFrame,
                           numeric_only: bool = True,
                           title: str = 'Feature Correlation Heatmap') -> plt.Figure:
    """
    Plot a correlation heatmap for numeric features.
    
    Args:
        df: DataFrame containing transaction data
        numeric_only: Whether to include only numeric columns
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    # Select numeric columns if required
    if numeric_only:
        data = df.select_dtypes(include=[np.number])
    else:
        data = df
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title(title, fontsize=14)
    
    return fig


def create_interactive_scatter(df: pd.DataFrame,
                              x_feature: str,
                              y_feature: str,
                              color_by: str = 'Is Fraudulent',
                              size_feature: Optional[str] = None,
                              title: Optional[str] = None) -> go.Figure:
    """
    Create an interactive scatter plot using Plotly.
    
    Args:
        df: DataFrame containing transaction data
        x_feature: Feature to plot on x-axis
        y_feature: Feature to plot on y-axis
        color_by: Feature to color points by
        size_feature: Feature to size points by
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Sample data if too large
    if len(df) > 10000:
        # Ensure we have representative samples of both classes
        fraud_sample = df[df[color_by] == 1].sample(min(5000, df[df[color_by] == 1].shape[0]))
        non_fraud_sample = df[df[color_by] == 0].sample(min(5000, df[df[color_by] == 0].shape[0]))
        plot_df = pd.concat([fraud_sample, non_fraud_sample])
    else:
        plot_df = df
    
    if title is None:
        title = f'{y_feature} vs {x_feature} Colored by {color_by}'
    
    # Create hover text
    hover_data = {
        x_feature: True,
        y_feature: True,
        color_by: True
    }
    
    fig = px.scatter(
        plot_df,
        x=x_feature,
        y=y_feature,
        color=color_by,
        size=size_feature,
        opacity=0.7,
        title=title,
        hover_data=hover_data,
        color_discrete_map={0: 'blue', 1: 'red'} if color_by == 'Is Fraudulent' else None
    )
    
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        legend_title=color_by,
        height=700,
        width=900
    )
    
    return fig


def create_interactive_histograms(df: pd.DataFrame,
                                feature: str,
                                target_col: str = 'Is Fraudulent',
                                title: Optional[str] = None) -> go.Figure:
    """
    Create interactive histograms using Plotly to compare distributions.
    
    Args:
        df: DataFrame containing transaction data
        feature: Feature to plot
        target_col: Target column for comparison
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if title is None:
        title = f'Distribution of {feature} by Fraud Status'
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    
    # Add histogram for non-fraudulent transactions
    fig.add_trace(
        go.Histogram(
            x=df[df[target_col] == 0][feature],
            name='Not Fraud',
            marker_color='blue',
            opacity=0.7,
            histnorm='probability'
        ),
        row=1, col=1
    )
    
    # Add histogram for fraudulent transactions
    fig.add_trace(
        go.Histogram(
            x=df[df[target_col] == 1][feature],
            name='Fraud',
            marker_color='red',
            opacity=0.7,
            histnorm='probability'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis2_title=feature,
        yaxis_title='Probability',
        yaxis2_title='Probability',
        height=600,
        width=900,
        bargap=0.1
    )
    
    return fig