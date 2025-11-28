"""
Example usage script for IDS ML project.
Demonstrates how to use the IDS ML system.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import DataPreprocessor
from src.models import RandomForestIDS, SVMIDS, NeuralNetworkIDS
from src.evaluation import ModelEvaluator


def main():
    """Example usage of the IDS ML system."""
    print("="*60)
    print("IDS ML - Example Usage")
    print("="*60)
    
    # Step 1: Create sample data if it doesn't exist
    data_path = 'data/raw/sample_data.csv'
    if not os.path.exists(data_path):
        print("\n1. Creating sample data...")
        from src.preprocessing import main as create_sample
        create_sample()
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)
    df_cleaned = preprocessor.clean_data(df)
    df_features = preprocessor.extract_features(df_cleaned)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df_features)
    
    # Step 3: Train a model (Random Forest as example)
    print("\n3. Training Random Forest model...")
    model = RandomForestIDS(n_estimators=50, random_state=42)  # Reduced for faster demo
    model.train(X_train, y_train)
    
    # Step 4: Evaluate model
    print("\n4. Evaluating model...")
    evaluator = ModelEvaluator("Random Forest")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)
    
    # Step 5: Save model
    print("\n5. Saving model...")
    os.makedirs('models', exist_ok=True)
    model.save('models/example_model.pkl')
    preprocessor.save_preprocessor('models/example_preprocessor.pkl')
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Use 'python scripts/train.py' to train all models")
    print("2. Use 'python scripts/predict.py --input <data_file>' for predictions")
    print("3. Open Jupyter notebooks in notebooks/ for detailed analysis")


if __name__ == "__main__":
    main()

