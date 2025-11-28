"""
Visualization utilities for IDS ML project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


class IDSVisualizer:
    """Visualization utilities for IDS data and results."""
    
    def __init__(self, style='seaborn-v0_8', figsize=(10, 6)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default style if seaborn style not available
            plt.style.use('default')
        self.figsize = figsize
        sns.set_palette("husl")
    
    def plot_class_distribution(self, df, target_column='label', save_path=None):
        """
        Plot the distribution of attack classes.
        
        Args:
            df: DataFrame with target column
            target_column: Name of the target column
            save_path: Path to save the figure
        """
        plt.figure(figsize=self.figsize)
        
        value_counts = df[target_column].value_counts()
        colors = sns.color_palette("husl", len(value_counts))
        
        plt.bar(range(len(value_counts)), value_counts.values, color=colors)
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        plt.xlabel('Attack Type')
        plt.ylabel('Count')
        plt.title('Distribution of Attack Classes')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_distributions(self, df, features=None, n_features=6, save_path=None):
        """
        Plot distributions of selected features.
        
        Args:
            df: DataFrame
            features: List of feature names to plot
            n_features: Number of features to plot if features is None
            save_path: Path to save the figure
        """
        if features is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = numeric_cols[:n_features].tolist()
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, feature in enumerate(features):
            if idx < len(axes):
                df[feature].hist(bins=50, ax=axes[idx], edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {feature}')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(features), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, df, features=None, save_path=None):
        """
        Plot correlation matrix of features.
        
        Args:
            df: DataFrame
            features: List of feature names
            save_path: Path to save the figure
        """
        if features is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = numeric_cols.tolist()
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[features].corr()
        
        sns.heatmap(
            correlation_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attack_patterns(self, df, feature1, feature2, target_column='label', save_path=None):
        """
        Plot attack patterns using two features.
        
        Args:
            df: DataFrame
            feature1: First feature name
            feature2: Second feature name
            target_column: Target column name
            save_path: Path to save the figure
        """
        plt.figure(figsize=self.figsize)
        
        unique_labels = df[target_column].unique()
        colors = sns.color_palette("husl", len(unique_labels))
        
        for idx, label in enumerate(unique_labels):
            mask = df[target_column] == label
            plt.scatter(
                df[mask][feature1],
                df[mask][feature2],
                label=label,
                alpha=0.6,
                s=20,
                color=colors[idx]
            )
        
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f'Attack Patterns: {feature1} vs {feature2}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_performance_comparison(self, results_dict, save_path=None):
        """
        Compare model performance metrics.
        
        Args:
            results_dict: Dictionary of {model_name: {metric: value}}
            save_path: Path to save the figure
        """
        df = pd.DataFrame(results_dict).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            if metric in df.columns:
                bars = ax.bar(range(len(df)), df[metric], color='steelblue', alpha=0.7, edgecolor='black')
                ax.set_xticks(range(len(df)))
                ax.set_xticklabels(df.index, rotation=45, ha='right')
                ax.set_ylabel('Score')
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot neural network training history.
        
        Args:
            history: Keras training history
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, df, target_column='label'):
        """
        Create an interactive Plotly dashboard.
        
        Args:
            df: DataFrame
            target_column: Target column name
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Class Distribution', 'Feature Correlation', 
                          'Attack Patterns', 'Feature Importance'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Class distribution
        value_counts = df[target_column].value_counts()
        fig.add_trace(
            go.Bar(x=value_counts.index, y=value_counts.values, name="Count"),
            row=1, col=1
        )
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
        corr_matrix = df[numeric_cols].corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(height=800, title_text="IDS Data Dashboard", showlegend=False)
        fig.show()

