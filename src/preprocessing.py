"""
Data preprocessing module for IDS ML project.
Handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    """Preprocess network traffic data for ML models."""
    
    def __init__(self, data_path=None):
        """
        Initialize the preprocessor.
        
        Args:
            data_path: Path to the raw data file
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_data(self, file_path):
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame: Loaded data
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.pkl'):
                df = pd.read_pickle(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or PKL.")
            
            print(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        print("Cleaning data...")
        
        # Remove duplicate rows
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        print(f"Removed {initial_shape - df.shape[0]} duplicate rows")
        
        # Handle missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Handling missing values...")
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown')
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        
        print(f"Cleaned data shape: {df.shape}")
        return df
    
    def extract_features(self, df):
        """
        Extract and engineer features from network traffic data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: DataFrame with engineered features
        """
        print("Extracting features...")
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if present
        target_cols = ['label', 'attack', 'class', 'target']
        for col in target_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)
            if col in categorical_cols:
                categorical_cols.remove(col)
        
        # Encode categorical variables
        df_encoded = df.copy()
        for col in categorical_cols:
            if col not in target_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                numeric_cols.append(col)
        
        self.feature_columns = numeric_cols
        print(f"Feature columns: {len(self.feature_columns)}")
        
        return df_encoded
    
    def prepare_data(self, df, target_column='label', test_size=0.2, random_state=42):
        """
        Prepare data for training by splitting and scaling.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Preparing data for training...")
        
        # Identify target column
        possible_targets = ['label', 'attack', 'class', 'target']
        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError("Target column not found. Expected one of: label, attack, class, target")
        
        # Separate features and target
        X = df[self.feature_columns].copy()
        y = df[target_col].copy()
        
        # Encode target labels
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=self.feature_columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=self.feature_columns, index=X_test.index)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Number of classes: {len(np.unique(y))}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, file_path='models/preprocessor.pkl'):
        """Save the preprocessor for later use."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }, file_path)
        print(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path='models/preprocessor.pkl'):
        """Load a saved preprocessor."""
        preprocessor = joblib.load(file_path)
        self.scaler = preprocessor['scaler']
        self.label_encoder = preprocessor['label_encoder']
        self.feature_columns = preprocessor['feature_columns']
        print(f"Preprocessor loaded from {file_path}")


def main():
    """Example usage of the preprocessor."""
    # Create sample data for demonstration
    print("Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'duration': np.random.exponential(10, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh'], n_samples),
        'src_bytes': np.random.poisson(1000, n_samples),
        'dst_bytes': np.random.poisson(2000, n_samples),
        'count': np.random.poisson(5, n_samples),
        'srv_count': np.random.poisson(3, n_samples),
        'dst_host_count': np.random.poisson(10, n_samples),
        'label': np.random.choice(['normal', 'dos', 'probe', 'r2l', 'u2r'], n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save sample data
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/sample_data.csv', index=False)
    print("Sample data saved to data/raw/sample_data.csv")
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.clean_data(df)
    df_features = preprocessor.extract_features(df_cleaned)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df_features)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    pd.Series(y_train).to_csv('data/processed/y_train.csv', index=False)
    pd.Series(y_test).to_csv('data/processed/y_test.csv', index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()

