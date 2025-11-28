"""
Model evaluation module for IDS.
Provides metrics, confusion matrix, and performance analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ModelEvaluator:
    """Evaluate ML models for IDS."""
    
    def __init__(self, model_name="Model"):
        self.model_name = model_name
        self.metrics = {}
        
    def evaluate(self, y_true, y_pred, y_proba=None):
        """
        Evaluate model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            dict: Dictionary of metrics
        """
        print(f"\n{'='*50}")
        print(f"Evaluating {self.model_name}")
        print(f"{'='*50}")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # ROC AUC if probabilities available
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten())
                else:
                    # Multi-class classification
                    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                self.metrics['roc_auc'] = auc
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")
        
        # Print metrics
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        if 'roc_auc' in self.metrics:
            print(f"ROC AUC:   {self.metrics['roc_auc']:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        return self.metrics
    
    def confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """
        Generate and display confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save the figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names if class_names else range(len(cm)),
            yticklabels=class_names if class_names else range(len(cm))
        )
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        return cm
    
    def roc_curve(self, y_true, y_proba, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save the figure
        """
        if len(np.unique(y_true)) != 2:
            print("ROC curve is only available for binary classification")
            return
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten())
        auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten())
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def precision_recall_curve(self, y_true, y_proba, save_path=None):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save the figure
        """
        if len(np.unique(y_true)) != 2:
            print("Precision-Recall curve is only available for binary classification")
            return
        
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, models_results, save_path=None):
        """
        Compare multiple models' performance.
        
        Args:
            models_results: Dictionary of {model_name: metrics_dict}
            save_path: Path to save the comparison figure
        """
        df = pd.DataFrame(models_results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            if metric in df.columns:
                df[metric].plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel('Score')
                ax.set_xlabel('Model')
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3, axis='y')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison saved to {save_path}")
        
        plt.show()
        
        return df


def evaluate_model(model, X_test, y_test, model_name="Model", class_names=None):
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        class_names: List of class names
        
    Returns:
        dict: Evaluation metrics
    """
    evaluator = ModelEvaluator(model_name)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Evaluate
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)
    
    # Generate confusion matrix
    evaluator.confusion_matrix(y_test, y_pred, class_names)
    
    return metrics, evaluator

