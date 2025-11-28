"""
Machine learning models for IDS.
Includes Random Forest, SVM, and Neural Network implementations.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os


class IDSModel:
    """Base class for IDS ML models."""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """Train the model."""
        raise NotImplementedError("Subclass must implement train method")
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            return None
    
    def save(self, file_path):
        """Save the model."""
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")
    
    def load(self, file_path):
        """Load a saved model."""
        self.model = joblib.load(file_path)
        self.is_trained = True
        print(f"Model loaded from {file_path}")


class RandomForestIDS(IDSModel):
    """Random Forest classifier for IDS."""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
    
    def train(self, X_train, y_train, tune_hyperparameters=False):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        print(f"Training {self.model_name}...")
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        print(f"{self.model_name} training completed!")


class SVMIDS(IDSModel):
    """Support Vector Machine classifier for IDS."""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=42):
        super().__init__("SVM")
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True,
            verbose=True
        )
    
    def train(self, X_train, y_train, tune_hyperparameters=False):
        """
        Train the SVM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        print(f"Training {self.model_name}...")
        
        # For large datasets, use a sample for SVM training
        if len(X_train) > 10000:
            print("Large dataset detected. Sampling 10000 samples for SVM training...")
            sample_idx = np.random.choice(len(X_train), 10000, replace=False)
            X_train_sample = X_train.iloc[sample_idx] if isinstance(X_train, pd.DataFrame) else X_train[sample_idx]
            y_train_sample = y_train[sample_idx] if isinstance(y_train, (pd.Series, np.ndarray)) else y_train.iloc[sample_idx]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_sample, y_train_sample)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_train_sample, y_train_sample)
        
        self.is_trained = True
        print(f"{self.model_name} training completed!")


class NeuralNetworkIDS(IDSModel):
    """Neural Network classifier for IDS using TensorFlow/Keras."""
    
    def __init__(self, input_dim=None, num_classes=None, hidden_layers=[128, 64], 
                 dropout_rate=0.3, learning_rate=0.001):
        super().__init__("Neural Network")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self, input_dim, num_classes):
        """Build the neural network architecture."""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(self.hidden_layers[0], activation='relu', input_dim=input_dim))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, verbose=1):
        """
        Train the Neural Network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
        """
        print(f"Training {self.model_name}...")
        
        # Convert to numpy arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values
        
        # Determine input and output dimensions
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(input_dim, num_classes)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, (pd.Series, pd.DataFrame)):
                y_val = y_val.values
            validation_data = (X_val, y_val)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        print(f"{self.model_name} training completed!")
    
    def predict(self, X):
        """Make predictions with the neural network."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = self.model.predict(X, verbose=0)
        
        # Convert probabilities to class predictions
        if predictions.shape[1] == 1:
            # Binary classification
            return (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X, verbose=0)
    
    def save(self, file_path):
        """Save the neural network model."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.model.save(file_path)
        print(f"Model saved to {file_path}")
    
    def load(self, file_path):
        """Load a saved neural network model."""
        self.model = keras.models.load_model(file_path)
        self.is_trained = True
        print(f"Model loaded from {file_path}")


def create_all_models():
    """Create instances of all IDS models."""
    return {
        'random_forest': RandomForestIDS(),
        'svm': SVMIDS(),
        'neural_network': NeuralNetworkIDS()
    }

