"""
Real-time prediction script for IDS.
Loads trained models and makes predictions on new data.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import DataPreprocessor
from src.models import RandomForestIDS, SVMIDS, NeuralNetworkIDS


def load_model(model_type, model_path):
    """
    Load a trained model.
    
    Args:
        model_type: Type of model ('random_forest', 'svm', 'neural_network')
        model_path: Path to the saved model
        
    Returns:
        Trained model object
    """
    if model_type == 'random_forest':
        model = RandomForestIDS()
    elif model_type == 'svm':
        model = SVMIDS()
    elif model_type == 'neural_network':
        model = NeuralNetworkIDS()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load(model_path)
    return model


def predict(model, preprocessor, data_path, output_path=None):
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        preprocessor: Data preprocessor
        data_path: Path to input data
        output_path: Path to save predictions (optional)
        
    Returns:
        DataFrame with predictions
    """
    print(f"Loading data from {data_path}...")
    df = preprocessor.load_data(data_path)
    
    if df is None:
        print("Error: Could not load data")
        return None
    
    print("Preprocessing data...")
    df_cleaned = preprocessor.clean_data(df)
    df_features = preprocessor.extract_features(df_cleaned)
    
    # Select features
    X = df_features[preprocessor.feature_columns].copy()
    
    # Scale features
    X_scaled = preprocessor.scaler.transform(X)
    X = pd.DataFrame(X_scaled, columns=preprocessor.feature_columns)
    
    print("Making predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Decode labels if needed
    if hasattr(preprocessor.label_encoder, 'classes_'):
        predicted_labels = preprocessor.label_encoder.inverse_transform(predictions)
    else:
        predicted_labels = predictions
    
    # Create results DataFrame
    results = df.copy()
    results['predicted_label'] = predicted_labels
    results['prediction_confidence'] = np.max(probabilities, axis=1) if probabilities is not None else None
    
    # Add probability for each class
    if probabilities is not None and hasattr(preprocessor.label_encoder, 'classes_'):
        for idx, class_name in enumerate(preprocessor.label_encoder.classes_):
            results[f'prob_{class_name}'] = probabilities[:, idx]
    
    # Print summary
    print("\n" + "="*60)
    print("Prediction Summary")
    print("="*60)
    print(f"Total samples: {len(results)}")
    print("\nPredicted class distribution:")
    print(results['predicted_label'].value_counts())
    
    # Identify potential attacks
    if 'normal' in results['predicted_label'].values:
        attacks = results[results['predicted_label'] != 'normal']
        if len(attacks) > 0:
            print(f"\n⚠️  Potential attacks detected: {len(attacks)}")
            print("\nAttack details:")
            print(attacks[['predicted_label', 'prediction_confidence']].head(10))
    
    # Save results
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Make predictions using trained IDS models')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'svm', 'neural_network'],
        help='Model to use for prediction (default: random_forest)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to saved model (default: models/{model}_model.pkl)'
    )
    parser.add_argument(
        '--preprocessor-path',
        type=str,
        default='models/preprocessor.pkl',
        help='Path to saved preprocessor (default: models/preprocessor.pkl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions (optional)'
    )
    
    args = parser.parse_args()
    
    # Set default model path
    if args.model_path is None:
        if args.model == 'neural_network':
            args.model_path = 'models/neural_network_model.h5'
        else:
            args.model_path = f'models/{args.model}_model.pkl'
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first using: python scripts/train.py")
        return
    
    if not os.path.exists(args.preprocessor_path):
        print(f"Error: Preprocessor file not found at {args.preprocessor_path}")
        print("Please train the model first using: python scripts/train.py")
        return
    
    # Load preprocessor
    print("Loading preprocessor...")
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor(args.preprocessor_path)
    
    # Load model
    print(f"Loading {args.model} model...")
    model = load_model(args.model, args.model_path)
    
    # Make predictions
    results = predict(model, preprocessor, args.input, args.output)
    
    if results is not None:
        print("\n✅ Prediction completed successfully!")


if __name__ == "__main__":
    main()

