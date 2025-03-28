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
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib as mpl
from matplotlib import rcParams
from cycler import cycler
from scipy.stats import gaussian_kde


def set_visualization_style() -> None:
    """
    Set up matplotlib and seaborn visualization style.
    """
    # Use ggplot style as base but customize further
    plt.style.use('ggplot')
    
    # Enable LaTeX rendering for text
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts} \usepackage{mathrsfs}'
    
    # Set seaborn aesthetics
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
    sns.set_context("notebook", font_scale=1.4)
    
    # Set default figure size and fonts
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    
    # Set larger font sizes
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    # Use scientific notation for large numbers
    plt.rcParams['axes.formatter.use_mathtext'] = True
    
    # Use professional color palettes
    plt.rcParams['axes.prop_cycle'] = cycler(color=[
        '#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', 
        '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9'
    ])
    
    # Set the background color to be slightly off-white for better contrast
    plt.rcParams['figure.facecolor'] = '#FAFAFA'
    plt.rcParams['axes.facecolor'] = '#FAFAFA'


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
    
    # Use professional colors from the color cycle
    colors = ['#0173B2', '#DE8F05']
    bars = ax.bar(
        ['Not Fraud', 'Fraud'],
        [fraud_counts[0], fraud_counts[1]],
        color=colors,
        width=0.6,
        edgecolor='black',
        linewidth=0.8
    )
    
    # Add data labels on top of bars with LaTeX formatting
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 5,
            f'${height:,}$',
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    # Add percentage of fraud to title with LaTeX formatting
    title_with_pct = f"{title}\n$\\textit{{Fraud: {fraud_percentage:.2f}\\% of transactions}}$"
    ax.set_title(title_with_pct, fontsize=14)
    ax.set_ylabel('$\\textbf{Count}$', fontsize=12)
    ax.set_ylim(0, fraud_counts[0] * 1.1)  # Add some padding for the labels
    
    # Add grid lines for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Better tick formatting with LaTeX
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Add text box with statistics using LaTeX formatting
    stats_text = (f"$\\textbf{{Total Transactions:}}$ ${len(df):,}$\n"
                  f"$\\textbf{{Non-Fraud:}}$ ${fraud_counts[0]:,}$ $({100 - fraud_percentage:.2f}\\%)$\n"
                  f"$\\textbf{{Fraud:}}$ ${fraud_counts[1]:,}$ $({fraud_percentage:.2f}\\%)$")
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)
    
    # Improve visual appearance with spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    # Add a light figure border
    fig.patch.set_linewidth(1)
    fig.patch.set_edgecolor('lightgray')
    
    plt.tight_layout()
    
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'hspace': 0.3})
    
    if title is None:
        title = f'Distribution of {feature} by Fraud Status'
    
    # Use custom colors from our professional palette
    non_fraud_color = '#0173B2'
    fraud_color = '#DE8F05'
    
    # Plot histogram with KDE for non-fraudulent transactions
    sns.histplot(
        df[df[target_col] == 0][feature],
        bins=bins,
        kde=True,
        color=non_fraud_color,
        alpha=0.6,
        ax=ax1,
        stat='probability',
        label='$\\textrm{Not Fraud}$',
        line_kws={'linewidth': 2}
    )
    ax1.set_title(f'${feature}$ Distribution - Not Fraud', fontsize=14)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot histogram with KDE for fraudulent transactions
    sns.histplot(
        df[df[target_col] == 1][feature],
        bins=bins,
        kde=True,
        color=fraud_color,
        alpha=0.6,
        ax=ax2,
        stat='probability',
        label='$\\textrm{Fraud}$',
        line_kws={'linewidth': 2}
    )
    ax2.set_title(f'${feature}$ Distribution - Fraud', fontsize=14)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis label with LaTeX formatting
    ax1.set_ylabel('$\\textrm{Probability Density}$', fontsize=12)
    ax2.set_ylabel('$\\textrm{Probability Density}$', fontsize=12)
    ax2.set_xlabel(f'${feature}$', fontsize=12)
    
    # Add descriptive statistics with LaTeX formatting
    for ax, fraud_val, label in zip([ax1, ax2], [0, 1], ['Not Fraud', 'Fraud']):
        subset = df[df[target_col] == fraud_val][feature]
        stats_text = (f"$\\boldsymbol{{\\mu}}$ = ${subset.mean():.2f}$\n"
                     f"$\\textrm{{Median}}$ = ${subset.median():.2f}$\n"
                     f"$\\boldsymbol{{\\sigma}}$ = ${subset.std():.2f}$\n"
                     f"$\\textrm{{Min}}$ = ${subset.min():.2f}$\n"
                     f"$\\textrm{{Max}}$ = ${subset.max():.2f}$")
        
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray')
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
              verticalalignment='top', bbox=props)
    
    # Format spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
    
    # Add figure title with LaTeX formatting
    fig.suptitle(f'$\\textbf{{{title}}}$', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate suptitle
    
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'hspace': 0.3})
    
    if title is None:
        title = f'Distribution of {feature} by Fraud Status'
    
    # Custom colors for better visualization
    non_fraud_color = '#0173B2'
    fraud_color = '#DE8F05'
    
    # Plot count of transactions by category and fraud status with improved styling
    count_data = pd.crosstab(df_plot[feature], df_plot[target_col])
    count_data.columns = ['Not Fraud', 'Fraud']
    count_data = count_data.reindex(fraud_rate.index)
    
    # Plot stacked bars for counts
    count_data.plot(kind='bar', stacked=True, ax=ax1, 
                    color=[non_fraud_color, fraud_color],
                    width=0.7, edgecolor='black', linewidth=0.5)
    
    # Format title and labels without LaTeX for category names
    ax1.set_title(f'Count of {feature} by Fraud Status', fontsize=14, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.legend(['Not Fraud', 'Fraud'], frameon=True, framealpha=0.9)
    
    # Add counts as text labels on the bars
    total_counts = count_data.sum(axis=1)
    for i, (idx, row) in enumerate(count_data.iterrows()):
        # Add total count at the top of the bar
        ax1.text(i, total_counts[i] + (total_counts.max() * 0.02), 
                f'{total_counts[i]:,}', 
                ha='center', va='bottom', fontsize=10)
        
    # Format y-axis with scientific notation if needed
    if total_counts.max() > 10000:
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Add grid for readability
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Sanitize labels to avoid LaTeX errors with special characters
    if max(len(str(label)) for label in fraud_rate.index) > 10:
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    else:
        plt.setp(ax1.get_xticklabels(), rotation=0)
    
    # Plot fraud rate by category with professional styling
    bars = ax2.bar(
        range(len(fraud_rate)), 
        fraud_rate.values * 100,  # Convert to percentage
        color='#029E73',
        width=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Format labels without LaTeX
    ax2.set_title(f'Fraud Rate by {feature}', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'{feature}', fontsize=12)
    ax2.set_ylabel('Fraud Rate (%)', fontsize=12)
    
    # Add percentage labels on the fraud rate bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.5,
            f'{height:.2f}%',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Set the same x-tick labels
    if max(len(str(label)) for label in fraud_rate.index) > 10:
        ax2.set_xticks(range(len(fraud_rate.index)))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    else:
        ax2.set_xticks(range(len(fraud_rate.index)))
        plt.setp(ax2.get_xticklabels(), rotation=0)
    
    # Make sure we're using the same labels for both axes
    ax2.set_xticklabels(fraud_rate.index)
    ax1.set_xticklabels(fraud_rate.index)
    
    # Add grid for readability
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal average line
    avg_fraud_rate = df[target_col].mean() * 100
    ax2.axhline(y=avg_fraud_rate, color='#D55E00', linestyle='--', alpha=0.8, 
                linewidth=1.5)
    ax2.text(
        len(fraud_rate) - 0.5, 
        avg_fraud_rate + 0.5, 
        f'Average: {avg_fraud_rate:.2f}%', 
        va='bottom', ha='right', 
        color='#D55E00', 
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0)
    )
    
    # Format spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
    
    # Add figure title without LaTeX formatting
    fig.suptitle(f'{title}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    return fig


def plot_time_patterns(df: pd.DataFrame,
                     date_col: str = 'Transaction Date',
                     target_col: str = 'Is Fraudulent',
                     title: str = 'Temporal Patterns of Fraud',
                     individual_plots: bool = False) -> Union[plt.Figure, List[plt.Figure]]:
    """
    Plot fraud rate by hour, day of week, and month.
    
    Args:
        df: DataFrame containing transaction data
        date_col: Name of the date column
        target_col: Name of the target column
        title: Plot title
        individual_plots: If True, return a list of separate figures instead of one combined figure
        
    Returns:
        Either a single Matplotlib figure object or a list of three figures
    """
    # Ensure date column is datetime type
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract time components
    df['hour'] = df[date_col].dt.hour
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    
    # Days of week names without LaTeX formatting
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Month names without LaTeX formatting
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Professional color palette
    primary_color = '#0173B2'
    secondary_color = '#D55E00'
    
    # Calculate data for all three plots
    hour_fraud = df.groupby('hour')[target_col].mean() * 100  # Convert to percentage
    hour_volume = df.groupby('hour').size() / len(df) * 100   # Convert to percentage
    
    day_fraud = df.groupby('day_of_week')[target_col].mean() * 100  # Convert to percentage
    day_volume = df.groupby('day_of_week').size() / len(df) * 100   # Convert to percentage
    
    month_fraud = df.groupby('month')[target_col].mean() * 100  # Convert to percentage
    month_volume = df.groupby('month').size() / len(df) * 100   # Convert to percentage
    
    if individual_plots:
        # Create three separate figures instead of one combined figure
        figures = []
        
        # Figure 1: Fraud rate by hour of day
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Fraud Rate (%)', color=primary_color, fontsize=12, fontweight='bold')
        
        # Plot with professional styling
        line1 = ax1.plot(hour_fraud.index, hour_fraud.values, 'o-', 
                        color=primary_color, linewidth=2, 
                        markersize=8, markeredgecolor='white', markeredgewidth=0.5)
        ax1.tick_params(axis='y', labelcolor=primary_color)
        ax1.set_title('Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add secondary axis for transaction volume with professional styling
        ax1_2 = ax1.twinx()
        ax1_2.set_ylabel('Transaction Volume (%)', color=secondary_color, fontsize=12, fontweight='bold')
        line2 = ax1_2.plot(hour_volume.index, hour_volume.values, 'o--', 
                          color=secondary_color, alpha=0.7, linewidth=1.5,
                          markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        ax1_2.tick_params(axis='y', labelcolor=secondary_color)
        
        # Add legend without LaTeX formatting
        lines = line1 + line2
        labels = ['Fraud Rate', 'Transaction Volume']
        ax1.legend(lines, labels, loc='upper right', frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        figures.append(fig1)
        
        # Figure 2: Fraud rate by day of week
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Fraud Rate (%)', color=primary_color, fontsize=12, fontweight='bold')
        line1 = ax2.plot(day_fraud.index, day_fraud.values, 'o-', 
                        color=primary_color, linewidth=2, 
                        markersize=8, markeredgecolor='white', markeredgewidth=0.5)
        ax2.tick_params(axis='y', labelcolor=primary_color)
        ax2.set_title('Fraud Rate by Day of Week', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(days)
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add secondary axis for transaction volume with professional styling
        ax2_2 = ax2.twinx()
        ax2_2.set_ylabel('Transaction Volume (%)', color=secondary_color, fontsize=12, fontweight='bold')
        line2 = ax2_2.plot(day_volume.index, day_volume.values, 'o--', 
                          color=secondary_color, alpha=0.7, linewidth=1.5,
                          markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        ax2_2.tick_params(axis='y', labelcolor=secondary_color)
        
        # Add legend
        lines = line1 + line2
        labels = ['Fraud Rate', 'Transaction Volume']
        ax2.legend(lines, labels, loc='upper right', frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        figures.append(fig2)
        
        # Figure 3: Fraud rate by month
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Fraud Rate (%)', color=primary_color, fontsize=12, fontweight='bold')
        line1 = ax3.plot(month_fraud.index, month_fraud.values, 'o-', 
                        color=primary_color, linewidth=2, 
                        markersize=8, markeredgecolor='white', markeredgewidth=0.5)
        ax3.tick_params(axis='y', labelcolor=primary_color)
        ax3.set_title('Fraud Rate by Month', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(months)
        ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add secondary axis for transaction volume with professional styling
        ax3_2 = ax3.twinx()
        ax3_2.set_ylabel('Transaction Volume (%)', color=secondary_color, fontsize=12, fontweight='bold')
        line2 = ax3_2.plot(month_volume.index, month_volume.values, 'o--', 
                          color=secondary_color, alpha=0.7, linewidth=1.5,
                          markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        ax3_2.tick_params(axis='y', labelcolor=secondary_color)
        
        # Add legend
        lines = line1 + line2
        labels = ['Fraud Rate', 'Transaction Volume']
        ax3.legend(lines, labels, loc='upper right', frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        figures.append(fig3)
        
        return figures
    
    else:
        # Create figure with three subplots - improved layout
        fig = plt.figure(figsize=(15, 21))  # Make taller rather than wider
        
        # Create grid spec for better control of subplot sizes
        gs = fig.add_gridspec(3, 1, hspace=0.3, height_ratios=[1, 1, 1])
        
        # Hour subplot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Fraud Rate (%)', color=primary_color, fontsize=12, fontweight='bold')
        
        # Plot with professional styling
        line1 = ax1.plot(hour_fraud.index, hour_fraud.values, 'o-', 
                        color=primary_color, linewidth=2, 
                        markersize=8, markeredgecolor='white', markeredgewidth=0.5)
        ax1.tick_params(axis='y', labelcolor=primary_color)
        ax1.set_title('Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add secondary axis for transaction volume with professional styling
        ax1_2 = ax1.twinx()
        ax1_2.set_ylabel('Transaction Volume (%)', color=secondary_color, fontsize=12, fontweight='bold')
        line2 = ax1_2.plot(hour_volume.index, hour_volume.values, 'o--', 
                          color=secondary_color, alpha=0.7, linewidth=1.5,
                          markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        ax1_2.tick_params(axis='y', labelcolor=secondary_color)
        
        # Add legend without LaTeX formatting
        lines = line1 + line2
        labels = ['Fraud Rate', 'Transaction Volume']
        ax1.legend(lines, labels, loc='upper right', frameon=True, framealpha=0.9)
        
        # Day of week subplot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Fraud Rate (%)', color=primary_color, fontsize=12, fontweight='bold')
        line1 = ax2.plot(day_fraud.index, day_fraud.values, 'o-', 
                        color=primary_color, linewidth=2, 
                        markersize=8, markeredgecolor='white', markeredgewidth=0.5)
        ax2.tick_params(axis='y', labelcolor=primary_color)
        ax2.set_title('Fraud Rate by Day of Week', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(days)
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add secondary axis for transaction volume with professional styling
        ax2_2 = ax2.twinx()
        ax2_2.set_ylabel('Transaction Volume (%)', color=secondary_color, fontsize=12, fontweight='bold')
        line2 = ax2_2.plot(day_volume.index, day_volume.values, 'o--', 
                          color=secondary_color, alpha=0.7, linewidth=1.5,
                          markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        ax2_2.tick_params(axis='y', labelcolor=secondary_color)
        
        # Add legend
        lines = line1 + line2
        labels = ['Fraud Rate', 'Transaction Volume']
        ax2.legend(lines, labels, loc='upper right', frameon=True, framealpha=0.9)
        
        # Month subplot
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Fraud Rate (%)', color=primary_color, fontsize=12, fontweight='bold')
        line1 = ax3.plot(month_fraud.index, month_fraud.values, 'o-', 
                        color=primary_color, linewidth=2, 
                        markersize=8, markeredgecolor='white', markeredgewidth=0.5)
        ax3.tick_params(axis='y', labelcolor=primary_color)
        ax3.set_title('Fraud Rate by Month', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(months)
        ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add secondary axis for transaction volume with professional styling
        ax3_2 = ax3.twinx()
        ax3_2.set_ylabel('Transaction Volume (%)', color=secondary_color, fontsize=12, fontweight='bold')
        line2 = ax3_2.plot(month_volume.index, month_volume.values, 'o--', 
                          color=secondary_color, alpha=0.7, linewidth=1.5,
                          markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        ax3_2.tick_params(axis='y', labelcolor=secondary_color)
        
        # Add legend
        lines = line1 + line2
        labels = ['Fraud Rate', 'Transaction Volume']
        ax3.legend(lines, labels, loc='upper right', frameon=True, framealpha=0.9)
        
        # Add figure title without LaTeX formatting
        fig.suptitle(f'{title}', fontsize=16, fontweight='bold', y=0.98)
        
        # Better tight_layout to use more of the figure space
        plt.tight_layout(rect=[0.03, 0.01, 0.97, 0.96])
        
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
    
    # Temporarily disable LaTeX rendering for the heatmap annotations only
    # We'll still use LaTeX for everything else
    plt.rcParams['text.usetex'] = True
    
    # Setup professional mask for better visualization
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Custom colormap with better contrast
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Create heatmap with enhanced styling but avoid LaTeX formatting for annotations
    with plt.rc_context({'text.usetex': True}):
        heatmap = sns.heatmap(
            corr_matrix,
            mask=mask,  # Show only lower triangular matrix
            annot=True,
            cmap=cmap,
            vmin=-1, vmax=1,
            center=0,
            fmt='.3f',
            linewidths=0.8,
            annot_kws={"size": 10},
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            ax=ax
        )
    
    # Add title with enhanced styling using LaTeX
    ax.set_title(f'$\\textbf{{{title}}}$', fontsize=16, pad=20)
    
    # Adjust margins for better appearance
    plt.tight_layout()
    
    # Add annotations for top correlations
    strong_pos_corr = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()[1:6]
    strong_neg_corr = corr_matrix.unstack().sort_values()[0:5]
    
    # Create annotation text with LaTeX formatting and proper escaping of special characters
    # Use a context manager to temporarily disable LaTeX
    with plt.rc_context({'text.usetex': True}):
        annotation_text = "Strongest Correlations:\n"
        for idx, val in strong_pos_corr.items():
            if idx[0] != idx[1]:  # Avoid self-correlations
                annotation_text += f"{idx[0]} and {idx[1]}: {val:.2f}\n"
        
        annotation_text += "\nStrongest Negative Correlations:\n"
        for idx, val in strong_neg_corr.items():
            if idx[0] != idx[1]:  # Avoid self-correlations
                annotation_text += f"{idx[0]} and {idx[1]}: {val:.2f}\n"
        
        # Add text box with statistics, positioned at top-right of frame
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
        plt.text(0.99, 0.99, annotation_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)
    # Re-enable LaTeX for axis labels, etc.
    plt.rcParams['text.usetex'] = True
    
    # Apply professional styling to axes with LaTeX formatting
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    # Set axis labels with LaTeX formatting
    ax.set_xlabel('$\\textbf{Features}$', fontsize=14)
    ax.set_ylabel('$\\textbf{Features}$', fontsize=14)
    
    # Add a descriptive caption at the bottom of the figure with LaTeX formatting
    caption_text = "$\\textit{Correlations between numeric features. Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation).}$"
    fig.text(0.5, 0.01, caption_text, ha='center', fontsize=10)
    
    # Add grid lines for better readability
    ax.grid(False)
    
    return fig


def create_interactive_scatter(df: pd.DataFrame,
                              x_feature: str,
                              y_feature: str,
                              color_by: str = 'Is Fraudulent',
                              size_feature: Optional[str] = None,
                              title: Optional[str] = None,
                              show_colorbar: bool = False) -> go.Figure:
    """
    Create an interactive scatter plot using Plotly.
    
    Args:
        df: DataFrame containing transaction data
        x_feature: Feature to plot on x-axis
        y_feature: Feature to plot on y-axis
        color_by: Feature to color points by
        size_feature: Feature to size points by
        title: Plot title
        show_colorbar: Whether to show the color bar (default: False)
        
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
    
    # Create hover text with more informative template
    hover_template = (
        f"<b>{x_feature}</b>: %{{x:.2f}}<br>" +
        f"<b>{y_feature}</b>: %{{y:.2f}}<br>" +
        f"<b>{color_by}</b>: %{{customdata}}<br>" +
        "<extra></extra>"  # Remove secondary hover info
    )
    
    # Custom colors for professional appearance
    colors = {
        0: '#0173B2',  # Blue for not fraud
        1: '#DE8F05'   # Orange for fraud
    } if color_by == 'Is Fraudulent' else None
    
    # Create figure with custom styling
    fig = px.scatter(
        plot_df,
        x=x_feature,
        y=y_feature,
        color=color_by,
        size=size_feature,
        opacity=0.75,
        custom_data=[plot_df[color_by]],  # Add custom data for hover template
        color_discrete_map=colors,
        template='plotly_white'  # Use a clean white template
    )
    
    # Update hover info
    fig.update_traces(
        hovertemplate=hover_template,
        marker=dict(
            line=dict(width=0.5, color='DarkSlateGrey')
        )
    )
    
    # Apply professional formatting without LaTeX
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'font': {'size': 22, 'family': 'Arial, sans-serif'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        # Axis labels without LaTeX
        xaxis_title=f"<b>{x_feature}</b>",
        yaxis_title=f"<b>{y_feature}</b>",
        legend_title=f"<b>{color_by}</b>",
        font=dict(
            family="Arial, sans-serif",
            size=14
        ),
        height=700,
        width=1000,
        # Add grid for better readability
        xaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='darkgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='darkgray'
        ),
        # Add light background color for better contrast
        paper_bgcolor='rgba(250,250,250,1)',
        plot_bgcolor='rgba(250,250,250,1)',
        # Better legend styling
        legend=dict(
            bordercolor="Black",
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.9)'
        ),
        # Hide color bar
        coloraxis_showscale=show_colorbar
    )
    
    # Hide color bar if requested
    if not show_colorbar:
        fig.update_coloraxes(showscale=False)
    
    # Add trend line for each class if we're coloring by fraud status
    if color_by == 'Is Fraudulent':
        for fraud_val in plot_df[color_by].unique():
            subset = plot_df[plot_df[color_by] == fraud_val]
            
            if len(subset) >= 10:  # Only add trendline if enough points
                fig.add_trace(
                    go.Scatter(
                        x=subset[x_feature],
                        y=subset[y_feature],
                        mode='lines',
                        name=f'Trend {"Fraud" if fraud_val == 1 else "Not Fraud"}',
                        line=dict(
                            color='darkred' if fraud_val == 1 else 'darkblue',
                            width=2,
                            dash='dash'
                        ),
                        opacity=0.7,
                        showlegend=True
                    )
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
    
    # Use professional colors
    non_fraud_color = '#0173B2'  # Blue for not fraud
    fraud_color = '#DE8F05'      # Orange for fraud
    
    # Prepare histogram data
    non_fraud_data = df[df[target_col] == 0][feature].dropna()
    fraud_data = df[df[target_col] == 1][feature].dropna()
    
    # Calculate statistics for annotations
    non_fraud_stats = {
        'mean': non_fraud_data.mean(),
        'median': non_fraud_data.median(),
        'std': non_fraud_data.std(),
        'count': len(non_fraud_data)
    }
    
    fraud_stats = {
        'mean': fraud_data.mean(),
        'median': fraud_data.median(),
        'std': fraud_data.std(),
        'count': len(fraud_data)
    }
    
    # Create subplots with better spacing
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            f"<b>{feature} Distribution - Not Fraud (n={non_fraud_stats['count']:,})</b>",
            f"<b>{feature} Distribution - Fraud (n={fraud_stats['count']:,})</b>"
        ]
    )
    
    # Add histogram with KDE for non-fraudulent transactions
    fig.add_trace(
        go.Histogram(
            x=non_fraud_data,
            name='Not Fraud',
            marker=dict(
                color=non_fraud_color,
                line=dict(color='rgba(0, 0, 0, 0.5)', width=0.6)
            ),
            opacity=0.75,
            histnorm='probability density',
            showlegend=False,
            hovertemplate=
                f"<b>{feature}</b>: %{{x}}<br>" +
                "<b>Density</b>: %{y:.4f}<br>" +
                "<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add KDE line for non-fraud
    x_range = np.linspace(non_fraud_data.min(), non_fraud_data.max(), 500)
    kde = gaussian_kde(non_fraud_data)
    y_kde = kde(x_range)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_kde,
            mode='lines',
            name='Not Fraud KDE',
            line=dict(color=non_fraud_color, width=2),
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # Add vertical lines for mean and median (non-fraud)
    fig.add_vline(x=non_fraud_stats['mean'], line=dict(color='rgba(0,0,0,0.7)', width=1, dash='dash'), 
                 row=1, col=1)
    fig.add_vline(x=non_fraud_stats['median'], line=dict(color='rgba(0,0,0,0.7)', width=1, dash='dot'), 
                 row=1, col=1)
    
    # Add histogram with KDE for fraudulent transactions
    fig.add_trace(
        go.Histogram(
            x=fraud_data,
            name='Fraud',
            marker=dict(
                color=fraud_color,
                line=dict(color='rgba(0, 0, 0, 0.5)', width=0.6)
            ),
            opacity=0.75,
            histnorm='probability density',
            showlegend=False,
            hovertemplate=
                f"<b>{feature}</b>: %{{x}}<br>" +
                "<b>Density</b>: %{y:.4f}<br>" +
                "<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Add KDE line for fraud
    x_range_fraud = np.linspace(fraud_data.min(), fraud_data.max(), 500)
    kde_fraud = gaussian_kde(fraud_data)
    y_kde_fraud = kde_fraud(x_range_fraud)
    fig.add_trace(
        go.Scatter(
            x=x_range_fraud,
            y=y_kde_fraud,
            mode='lines',
            name='Fraud KDE',
            line=dict(color=fraud_color, width=2),
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    # Add vertical lines for mean and median (fraud)
    fig.add_vline(x=fraud_stats['mean'], line=dict(color='rgba(0,0,0,0.7)', width=1, dash='dash'), 
                 row=2, col=1)
    fig.add_vline(x=fraud_stats['median'], line=dict(color='rgba(0,0,0,0.7)', width=1, dash='dot'), 
                 row=2, col=1)
    
    # Add annotations for statistics (non-fraud)
    non_fraud_annotations = [
        dict(
            x=0.02, y=0.85,
            xref="paper", yref="paper",
            text=f"<b>Mean:</b> {non_fraud_stats['mean']:.2f}<br>" +
                 f"<b>Median:</b> {non_fraud_stats['median']:.2f}<br>" +
                 f"<b>Std Dev:</b> {non_fraud_stats['std']:.2f}",
            showarrow=False,
            font=dict(family="Arial, sans-serif", size=12, color="#000000"),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            borderpad=6
        )
    ]
    
    # Add annotations for statistics (fraud)
    fraud_annotations = [
        dict(
            x=0.02, y=0.35,
            xref="paper", yref="paper",
            text=f"<b>Mean:</b> {fraud_stats['mean']:.2f}<br>" +
                 f"<b>Median:</b> {fraud_stats['median']:.2f}<br>" +
                 f"<b>Std Dev:</b> {fraud_stats['std']:.2f}",
            showarrow=False,
            font=dict(family="Arial, sans-serif", size=12, color="#000000"),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            borderpad=6
        )
    ]
    
    # Add legend for mean and median lines
    fig.add_trace(
        go.Scatter(
            x=[None], 
            y=[None],
            mode='lines',
            name='Mean',
            line=dict(color='black', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[None], 
            y=[None],
            mode='lines',
            name='Median',
            line=dict(color='black', width=1, dash='dot')
        ),
        row=1, col=1
    )
    
    # Update layout with professional styling
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'font': {'size': 24, 'family': 'Arial, sans-serif'},
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis2_title=f"<b>{feature}</b>",
        yaxis_title="<b>Probability Density</b>",
        yaxis2_title="<b>Probability Density</b>",
        height=700,
        width=1000,
        bargap=0.1,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        annotations=non_fraud_annotations + fraud_annotations,
        margin=dict(l=80, r=50, t=100, b=80),
        paper_bgcolor='rgba(250,250,250,1)',
        plot_bgcolor='rgba(250,250,250,1)',
    )
    
    # Update axes styling for more professional appearance
    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='darkgray',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='darkgray',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    return fig