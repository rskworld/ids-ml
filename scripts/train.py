"""
Training script for IDS ML models.
Trains Random Forest, SVM, and Neural Network models.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import DataPreprocessor
from src.models import RandomForestIDS, SVMIDS, NeuralNetworkIDS
from src.evaluation import ModelEvaluator


def train_all_models(data_path, test_size=0.2, random_state=42):
    """
    Train all IDS models.
    
    Args:
        data_path: Path to the dataset
        test_size: Proportion of test set
        random_state: Random seed
    """
    print("="*60)
    print("IDS ML Model Training")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    print("\n1. Loading data...")
    df = preprocessor.load_data(data_path)
    
    if df is None:
        print("Error: Could not load data")
        return
    
    print("\n2. Cleaning data...")
    df_cleaned = preprocessor.clean_data(df)
    
    print("\n3. Extracting features...")
    df_features = preprocessor.extract_features(df_cleaned)
    
    print("\n4. Preparing training and test sets...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        df_features, test_size=test_size, random_state=random_state
    )
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Initialize models
    models = {
        'Random Forest': RandomForestIDS(n_estimators=100, random_state=random_state),
        'SVM': SVMIDS(kernel='rbf', C=1.0, random_state=random_state),
        'Neural Network': NeuralNetworkIDS(hidden_layers=[128, 64], dropout_rate=0.3)
    }
    
    # Train models and collect results
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        try:
            if model_name == 'Neural Network':
                # Split training data for validation
                from sklearn.model_selection import train_test_split
                X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=random_state
                )
                model.train(X_train_nn, y_train_nn, X_val_nn, y_val_nn, epochs=50, batch_size=32)
            else:
                model.train(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Evaluate model
            evaluator = ModelEvaluator(model_name)
            metrics = evaluator.evaluate(y_test, y_pred, y_proba)
            results[model_name] = metrics
            
            # Save model
            model_path = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
            if model_name == 'Neural Network':
                model.save(f'models/{model_name.lower().replace(" ", "_")}_model.h5')
            else:
                model.save(model_path)
            
            # Generate confusion matrix
            class_names = preprocessor.label_encoder.classes_ if hasattr(preprocessor.label_encoder, 'classes_') else None
            evaluator.confusion_matrix(
                y_test, y_pred, 
                class_names=class_names,
                save_path=f'models/{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
            )
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare models
    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}")
    
    if results:
        comparison_df = pd.DataFrame(results).T
        print("\nPerformance Metrics:")
        print(comparison_df)
        
        # Save comparison
        comparison_df.to_csv('models/model_comparison.csv')
        
        # Visualize comparison
        from src.visualization import IDSVisualizer
        visualizer = IDSVisualizer()
        visualizer.plot_model_performance_comparison(
            results,
            save_path='models/model_comparison.png'
        )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train IDS ML models')
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/sample_data.csv',
        help='Path to the dataset (default: data/raw/sample_data.csv)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of test set (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Check if data file exists, if not create sample data
    if not os.path.exists(args.data):
        print(f"Data file not found at {args.data}")
        print("Creating sample data...")
        from src.preprocessing import main as create_sample
        create_sample()
        args.data = 'data/raw/sample_data.csv'
    
    # Train models
    train_all_models(
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()

