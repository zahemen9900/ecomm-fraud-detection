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
    #check if all items for said feature is string
    if any(isinstance(x, str) for x in df[feature]):
        df[feature] = df[feature].str.replace("&", "\&", regex=False) # Escape ampersands
        df[feature] = df[feature].str.replace("%", "\%", regex=False)  # Escape percent signs
        df[feature] = df[feature].str.replace("$", "\$", regex=False)  # Escape dollar signs

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
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
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


def plot_feature_importance(model, feature_names, title='Feature Importance Analysis', top_n=20):
    """
    Visualize feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute (e.g., RandomForest, XGBoost)
        feature_names: List of feature names
        title: Plot title
        top_n: Number of top features to display
        
    Returns:
        Matplotlib figure object
    """
    # Get feature importances
    try:
        # Try for models that have feature_importances_ (like tree-based models)
        importances = model.feature_importances_
    except AttributeError:
        try:
            # Try for models that have coef_ (like linear models)
            importances = np.abs(model.coef_[0])
        except:
            raise ValueError("Model doesn't have feature_importances_ or interpretable coefficients")
    
    # Create DataFrame with feature names and importance values
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    # Sort by importance and get top N features
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
    
    # Create plot with professional styling
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use custom color palette
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(feature_importance)))
    
    # Create horizontal bar chart
    bars = ax.barh(
        feature_importance['Feature'], 
        feature_importance['Importance'],
        color=colors,
        edgecolor='black',
        linewidth=0.6,
        height=0.7
    )
    
    # Add data labels to the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.001,  # Slight offset
            bar.get_y() + bar.get_height()/2, 
            f'{width:.4f}',
            ha='left', 
            va='center',
            fontsize=10
        )
    
    # Add a vertical line for the mean importance
    mean_importance = feature_importance['Importance'].mean()
    ax.axvline(x=mean_importance, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(
        mean_importance + 0.001, 
        -0.5, 
        f'Mean: {mean_importance:.4f}', 
        color='red', 
        fontsize=10,
        ha='left',
        va='top'
    )
    
    # Set title and axis labels with LaTeX formatting
    ax.set_title(f'$\\textbf{{{title}}}$', fontsize=16, pad=20)
    ax.set_xlabel('$\\textbf{Importance Score}$', fontsize=14)
    ax.set_ylabel('$\\textbf{Features}$', fontsize=14)
    
    # Set y-axis to display feature names in reverse order
    ax.invert_yaxis()
    
    # Add grid for better readability
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add plot border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    # Add information about what feature importance means
    info_text = ("$\\textit{Higher values indicate greater importance}$\n"
                "$\\textit{in the model's decision making process.}$")
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    return fig


def plot_pca_explained_variance(df, n_components=10, title='PCA Explained Variance Analysis'):
    """
    Visualize explained variance from Principal Component Analysis (PCA).
    
    Args:
        df: DataFrame containing numeric features for PCA
        n_components: Number of principal components to compute
        title: Plot title
        
    Returns:
        Tuple of (Matplotlib figure object, PCA object)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Preprocess data for PCA
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, min(numeric_df.shape)))
    pca.fit(scaled_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    
    # Plot explained variance
    explained_variance = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(explained_variance)
    components = range(1, len(explained_variance) + 1)
    
    # Plot individual explained variance
    ax1.bar(
        components, 
        explained_variance, 
        color=plt.cm.viridis(np.linspace(0, 0.9, len(explained_variance))),
        edgecolor='black',
        linewidth=0.6,
        alpha=0.8
    )
    
    # Add data labels to bars
    for i, v in enumerate(explained_variance):
        if v >= 1.0:  # Only label bars with significant variance
            ax1.text(
                i+1, 
                v + 0.5, 
                f'{v:.1f}%', 
                ha='center', 
                va='bottom',
                fontsize=10
            )
    
    ax1.set_title('$\\textbf{Explained Variance by Principal Components}$', fontsize=14)
    ax1.set_xlabel('$\\textbf{Principal Component}$', fontsize=12)
    ax1.set_ylabel('$\\textbf{Explained Variance (\%)}$', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Set x-axis to show integer ticks
    ax1.set_xticks(components)
    
    # Plot cumulative explained variance
    ax2.plot(
        components, 
        cumulative_variance, 
        'o-', 
        color='#0173B2', 
        linewidth=2.5,
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2,
        markeredgecolor='#0173B2'
    )
    
    # Add horizontal lines at key thresholds
    thresholds = [70, 80, 90, 95]
    threshold_colors = ['#D55E00', '#CC78BC', '#029E73', '#DE8F05'] 
    
    for threshold, color in zip(thresholds, threshold_colors):
        ax2.axhline(y=threshold, color=color, linestyle='--', alpha=0.7)
        
        # Find the first component that exceeds the threshold
        try:
            component_idx = next(i for i, val in enumerate(cumulative_variance) if val >= threshold)
            # Mark this point on the plot
            ax2.plot(
                component_idx + 1, 
                cumulative_variance[component_idx], 
                'o', 
                color=color, 
                markersize=8,
                markeredgecolor='black',
                markeredgewidth=1
            )
            # Add annotation
            ax2.annotate(
                f'{threshold}% at PC{component_idx + 1}',
                xy=(component_idx + 1, cumulative_variance[component_idx]),
                xytext=(component_idx + 1 - 0.3, cumulative_variance[component_idx] + 5),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color)
            )
        except StopIteration:
            # Threshold not reached
            pass
    
    # Add data labels at key points
    for i in range(0, len(cumulative_variance), max(1, len(cumulative_variance) // 5)):
        ax2.text(
            i+1, 
            cumulative_variance[i] + 2, 
            f'{cumulative_variance[i]:.1f}%', 
            ha='center', 
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='gray')
        )
    
    ax2.set_title('$\\textbf{Cumulative Explained Variance}$', fontsize=14)
    ax2.set_xlabel('$\\textbf{Number of Principal Components}$', fontsize=12)
    ax2.set_ylabel('$\\textbf{Cumulative Explained Variance (\%)}$', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 102)  # Leave room for annotations
    
    # Set x-axis to show integer ticks
    ax2.set_xticks(components)
    
    # Set the overall title
    plt.suptitle(f'$\\textbf{{{title}}}$', fontsize=16, y=0.98)
    
    # Add informative text
    info_text = ("$\\textit{PCA reduces dimensionality while preserving maximum variance.}$\n"
                "$\\textit{Select the number of components based on desired variance retention.}$")
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=12)
    
    # Format spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    return fig, pca


def plot_pca_components(df, target_col='Is Fraudulent', n_components=2, title='PCA Components Visualization'):
    """
    Create a 2D scatter plot of the first two PCA components, colored by fraud status.
    
    Args:
        df: DataFrame containing transaction data
        target_col: Name of the target column
        n_components: Number of principal components to compute 
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Preprocess data for PCA
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, min(X.shape)))
    X_pca = pca.fit_transform(X_scaled)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot data points colored by fraud status
    for label, color, marker in zip([0, 1], ['#0173B2', '#DE8F05'], ['o', 'X']):
        mask = (y == label)
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=f'{"Fraud" if label == 1 else "Not Fraud"}',
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
            s=70,
            marker=marker
        )
    
    # Add decision boundary ellipses (if classes are separable)
    from matplotlib.patches import Ellipse
    
    for label, color in zip([0, 1], ['#0173B2', '#DE8F05']):
        mask = (y == label)
        if np.sum(mask) > 2:  # Need at least 3 points to define an ellipse
            # Calculate mean and covariance of PCA coordinates for this class
            mean = np.mean(X_pca[mask, 0:2], axis=0)
            cov = np.cov(X_pca[mask, 0:2], rowvar=False)
            
            # Calculate eigenvalues and eigenvectors for the ellipse
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Create ellipse (95% confidence interval = 2.447 standard deviations)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width, height = 2.447 * np.sqrt(eigvals)
            
            # Create and add the ellipse to the plot
            ellipse = Ellipse(
                xy=mean, 
                width=width * 2, 
                height=height * 2,
                angle=angle,
                edgecolor=color,
                facecolor='none',
                linestyle='--',
                linewidth=2,
                alpha=0.7
            )
            ax.add_patch(ellipse)
    
    # Add title and labels with LaTeX formatting
    plt.title(f'$\\textbf{{{title}}}$', fontsize=16, pad=20)
    plt.xlabel(f'$\\textbf{{Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}\%)}}$', fontsize=14)
    plt.ylabel(f'$\\textbf{{Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}\%)}}$', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    legend = plt.legend(
        title='$\\textbf{Transaction Type}$',
        fontsize=12,
        title_fontsize=14,
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )
    
    # Add explained variance info
    explained_var_text = (
        f"$\\textbf{{Total explained variance: }}$"
        f"${(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]) * 100:.1f}\\%$"
    )
    plt.text(
        0.05, 0.05, 
        explained_var_text, 
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
    )
    
    # Format spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    # Add feature contribution text
    top_features = []
    for i, component in enumerate([0, 1]):
        # Get the top 5 features with highest absolute loadings for this component
        idx = np.argsort(np.abs(pca.components_[component]))[-5:]
        features = [(X.columns[j], pca.components_[component, j]) for j in idx]
        # Sort by actual (not absolute) loading to preserve sign
        features.sort(key=lambda x: -x[1])
        
        feature_text = f"$\\textbf{{PC{component+1} top features:}}$\n"
        for name, loading in features:
            feature_text += f"${name}: {loading:.3f}$\n"
        
        # Position text in opposite corners
        x_pos = 0.05 if i == 0 else 0.75
        y_pos = 0.95 if i == 0 else 0.95
        plt.text(
            x_pos, y_pos, 
            feature_text, 
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
        )
    
    plt.tight_layout()
    
    return fig


def plot_cluster_analysis(df, target_col='Is Fraudulent', n_clusters=3, title='Cluster Analysis of Transactions'):
    """
    Perform k-means clustering on the data and visualize the clusters.
    
    Args:
        df: DataFrame containing transaction data
        target_col: Name of the target column
        n_clusters: Number of clusters to form
        title: Plot title
        
    Returns:
        Tuple of (Matplotlib figure object, KMeans model, PCA model)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    # Preprocess data for clustering
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Use PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Define color maps
    fraud_colors = {0: '#0173B2', 1: '#DE8F05'}
    cluster_cmap = plt.cm.viridis  # Color map for clusters
    cluster_colors = cluster_cmap(np.linspace(0, 1, n_clusters))
    
    # Plot 1: Colored by fraud status
    for label, color in fraud_colors.items():
        mask = (y == label)
        ax1.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=f'{"Fraud" if label == 1 else "Not Fraud"}',
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
            s=70
        )
    
    ax1.set_title('$\\textbf{PCA Projection - Colored by Fraud Status}$', fontsize=14)
    ax1.set_xlabel(f'$\\textbf{{Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}\%)}}$', fontsize=12)
    ax1.set_ylabel(f'$\\textbf{{Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}\%)}}$', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(
        title='$\\textbf{Transaction Type}$',
        fontsize=11,
        title_fontsize=12,
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )
    
    # Plot 2: Colored by cluster
    for cluster_id in range(n_clusters):
        mask = (clusters == cluster_id)
        ax2.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[cluster_colors[cluster_id]],
            label=f'Cluster {cluster_id+1}',
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
            s=70
        )
    
    # Add cluster centers in PCA space
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax2.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c='red',
        marker='X',
        s=200,
        alpha=1,
        label='Cluster Centers',
        edgecolors='black',
        linewidth=1.5
    )
    
    ax2.set_title('$\\textbf{PCA Projection - Colored by Cluster}$', fontsize=14)
    ax2.set_xlabel(f'$\\textbf{{Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}\%)}}$', fontsize=12)
    ax2.set_ylabel(f'$\\textbf{{Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}\%)}}$', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(
        title='$\\textbf{Clusters}$',
        fontsize=11,
        title_fontsize=12,
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )
    
    # Calculate fraud rate per cluster
    cluster_fraud_rates = {}
    cluster_counts = {}
    for cluster_id in range(n_clusters):
        mask = (clusters == cluster_id)
        fraud_rate = y[mask].mean() * 100  # Convert to percentage
        count = np.sum(mask)
        cluster_fraud_rates[cluster_id] = fraud_rate
        cluster_counts[cluster_id] = count
    
    # Add cluster statistics text
    stats_text = "$\\textbf{Cluster Statistics:}$\n"
    for cluster_id in range(n_clusters):
        fraud_rate = cluster_fraud_rates[cluster_id]
        count = cluster_counts[cluster_id]
        percent = count / len(y) * 100
        stats_text += (f"$\\textbf{{Cluster {cluster_id+1}:}}$ Size: ${count}$ (${percent:.1f}\\%$), "
                      f"Fraud Rate: ${fraud_rate:.2f}\\%$\n")
    
    # Add as a text box under the second plot
    fig.text(
        0.5, 0.01, 
        stats_text, 
        ha='center', 
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
    )
    
    # Format spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
    
    # Set the overall title
    plt.suptitle(f'$\\textbf{{{title}}}$', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    return fig, kmeans, pca


def plot_biplot(df, target_col='Is Fraudulent', n_features=8, title='PCA Biplot - Features and Observations'):
    """
    Create a biplot showing both the PCA-transformed data points and feature vectors.
    
    Args:
        df: DataFrame containing transaction data
        target_col: Name of the target column
        n_features: Number of top features to display
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Preprocess data for PCA
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Scatter plot of data points
    for label, color, marker in zip([0, 1], ['#0173B2', '#DE8F05'], ['o', 'X']):
        mask = (y == label)
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=f'{"Fraud" if label == 1 else "Not Fraud"}',
            alpha=0.6,
            edgecolors='white',
            linewidth=0.5,
            s=60,
            marker=marker
        )
    
    # Get the scaling factor for the feature vectors
    # (Scale feature vectors to fit within the plot)
    pca_components = pca.components_
    feature_names = X.columns
    
    # Calculate the scaling factor based on the spread of the data
    scale_factor = np.max([np.abs(X_pca[:, 0]).max(), np.abs(X_pca[:, 1]).max()]) * 0.8
    
    # Calculate feature vector magnitudes and sort them
    feature_magnitudes = np.sqrt(pca_components[0]**2 + pca_components[1]**2)
    top_feature_idx = np.argsort(feature_magnitudes)[-n_features:]
    
    # Plot feature vectors
    for i in top_feature_idx:
        # Scale the feature vector
        x_coord = pca_components[0, i] * scale_factor
        y_coord = pca_components[1, i] * scale_factor
        
        # Plot vector
        plt.arrow(
            0, 0,
            x_coord, y_coord,
            color='darkred',
            alpha=0.8,
            width=0.01 * scale_factor,
            head_width=0.05 * scale_factor,
            head_length=0.08 * scale_factor,
            length_includes_head=True
        )
        
        # Add feature name label
        # Adjust text position to prevent overlap
        text_x = x_coord* 1.1
        text_y = y_coord * 1.1
        
        # Make text color match vector direction (red for positive, blue for negative)
        if x_coord**2 + y_coord**2 > 0:
            # Create background for text
            plt.text(
                text_x, text_y,
                feature_names[i],
                color='black',
                ha='center', va='center',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkred', boxstyle='round,pad=0.2')
            )
    
    # Add title and labels with LaTeX formatting
    plt.title(f'$\\textbf{{{title}}}$', fontsize=16, pad=20)
    plt.xlabel(f'$\\textbf{{Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}\%)}}$', fontsize=14)
    plt.ylabel(f'$\\textbf{{Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}\%)}}$', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    legend = plt.legend(
        title='$\\textbf{Transaction Type}$',
        fontsize=12,
        title_fontsize=14,
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )
    
    # Add explained variance info
    explained_var_text = (
        f"$\\textbf{{Total explained variance: }}$"
        f"${(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]) * 100:.1f}\\%$"
    )
    plt.text(
        0.05, 0.05, 
        explained_var_text, 
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
    )
    
    # Format spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    # Add explanatory text about the biplot
    info_text = ("$\\textit{A biplot shows both observations (points) and features (vectors) in the PCA space.}$\n"
                 "$\\textit{Feature vectors point in the direction of increasing values, with longer vectors having more influence.}$")
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    return fig


def plot_fraud_probability_distribution(y_true, y_proba, n_bins=50, threshold=0.5, 
                                       title='Fraud Probability Distribution'):
    """
    Visualize the distribution of predicted fraud probabilities.
    
    Args:
        y_true: True labels (0 for not fraud, 1 for fraud)
        y_proba: Predicted fraud probabilities from model
        n_bins: Number of bins for histogram
        threshold: Decision threshold for classification
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Split probabilities by actual class
    fraud_probs = y_proba[y_true == 1]
    non_fraud_probs = y_proba[y_true == 0]
    
    # Professional color palette
    non_fraud_color = '#0173B2'  # Blue
    fraud_color = '#DE8F05'      # Orange
    
    # Plot histograms with KDE
    bins = np.linspace(0, 1, n_bins)
    
    # Non-fraud distribution
    ax.hist(
        non_fraud_probs,
        bins=bins,
        alpha=0.6,
        color=non_fraud_color,
        edgecolor='black',
        linewidth=0.5,
        density=True,
        label='Not Fraud'
    )
    
    # Fraud distribution
    ax.hist(
        fraud_probs,
        bins=bins,
        alpha=0.6,
        color=fraud_color,
        edgecolor='black',
        linewidth=0.5,
        density=True,
        label='Fraud'
    )
    
    # Add KDE curves
    if len(non_fraud_probs) > 1:
        x_grid = np.linspace(0, 1, 1000)
        non_fraud_kde = gaussian_kde(non_fraud_probs)
        ax.plot(
            x_grid,
            non_fraud_kde(x_grid),
            color=non_fraud_color,
            linewidth=2,
            label='_nolegend_'
        )
    
    if len(fraud_probs) > 1:
        x_grid = np.linspace(0, 1, 1000)
        fraud_kde = gaussian_kde(fraud_probs)
        ax.plot(
            x_grid,
            fraud_kde(x_grid),
            color=fraud_color,
            linewidth=2,
            label='_nolegend_'
        )
    
    # Add vertical line for threshold
    ax.axvline(
        x=threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        alpha=0.8,
        label=f'Threshold: {threshold}'
    )
    
    # Add title and labels with LaTeX formatting
    ax.set_title(f'$\\textbf{{{title}}}$', fontsize=16, pad=20)
    ax.set_xlabel('$\\textbf{Predicted Fraud Probability}$', fontsize=14)
    ax.set_ylabel('$\\textbf{Density}$', fontsize=14)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(
        fontsize=12,
        loc='upper center',
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )
    
    # Calculate statistics for annotation
    fn_rate = np.mean((y_proba < threshold) & (y_true == 1)) * 100
    fp_rate = np.mean((y_proba >= threshold) & (y_true == 0)) * 100
    
    # Add statistics text box
    stats_text = (
        f"$\\textbf{{Model Performance:}}$\n"
        f"$\\textit{{False Negative Rate: }}{fn_rate:.2f}\\%$\n"
        f"$\\textit{{False Positive Rate: }}{fp_rate:.2f}\\%$\n"
        f"$\\textit{{True Fraud: }}{np.sum(y_true == 1)}$ $({np.mean(y_true == 1) * 100:.2f}\\%)$\n"
        f"$\\textit{{Predicted Fraud: }}{np.sum(y_proba >= threshold)}$ $({np.mean(y_proba >= threshold) * 100:.2f}\\%)$"
    )
    
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    # Add shaded regions for false positives and false negatives
    # False negatives (fraud classified as non-fraud)
    ax.fill_between(
        bins,
        0,
        5,  # Arbitrary height, will be clipped by plot limits
        where=(bins < threshold),
        color=fraud_color,
        alpha=0.2,
        label='_nolegend_'
    )
    ax.text(
        threshold/2, 
        ax.get_ylim()[1] * 0.9,
        'False Negatives',
        ha='center',
        va='top',
        fontsize=10,
        color=fraud_color,
        fontweight='bold'
    )
    
    # False positives (non-fraud classified as fraud)
    ax.fill_between(
        bins,
        0,
        5,  # Arbitrary height, will be clipped by plot limits
        where=(bins >= threshold),
        color=non_fraud_color,
        alpha=0.2,
        label='_nolegend_'
    )
    ax.text(
        (1 + threshold)/2, 
        ax.get_ylim()[1] * 0.9,
        'False Positives',
        ha='center',
        va='top',
        fontsize=10,
        color=non_fraud_color,
        fontweight='bold'
    )
    
    # Format spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    # Add explanatory text
    info_text = ("$\\textit{This plot shows the distribution of predicted fraud probabilities for actual fraud and non-fraud transactions.}$\n"
                 "$\\textit{The vertical line represents the classification threshold.}$")
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    return fig