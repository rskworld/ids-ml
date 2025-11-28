"""
Flask web application for IDS ML project.
Provides web interface and API endpoints for intrusion detection.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import DataPreprocessor
from src.models import RandomForestIDS, SVMIDS, NeuralNetworkIDS
from src.evaluation import ModelEvaluator

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variables for models and preprocessor
preprocessor = None
models = {}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_models():
    """Load trained models if they exist."""
    global preprocessor, models
    
    preprocessor_path = 'models/preprocessor.pkl'
    if os.path.exists(preprocessor_path):
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor(preprocessor_path)
        
        # Try to load models
        model_paths = {
            'random_forest': 'models/random_forest_model.pkl',
            'svm': 'models/svm_model.pkl',
            'neural_network': 'models/neural_network_model.h5'
        }
        
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                try:
                    if model_name == 'random_forest':
                        model = RandomForestIDS()
                        model.load(model_path)
                        models[model_name] = model
                    elif model_name == 'svm':
                        model = SVMIDS()
                        model.load(model_path)
                        models[model_name] = model
                    elif model_name == 'neural_network':
                        model = NeuralNetworkIDS()
                        model.load(model_path)
                        models[model_name] = model
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")


# Load models on startup
load_models()


@app.route('/')
def index():
    """Home page."""
    return app.send_static_file('index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    try:
        return app.send_static_file(filename)
    except:
        # If file not found, try as HTML
        if not filename.endswith('.html'):
            return app.send_static_file(filename + '.html')
        return "File not found", 404


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        model_name = request.form.get('model', 'random_forest')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and preprocess data
        if preprocessor is None:
            return jsonify({'error': 'Preprocessor not loaded. Please train models first.'}), 500
        
        df = preprocessor.load_data(filepath)
        if df is None:
            return jsonify({'error': 'Could not load data file'}), 400
        
        df_cleaned = preprocessor.clean_data(df)
        df_features = preprocessor.extract_features(df_cleaned)
        
        # Select features
        X = df_features[preprocessor.feature_columns].copy()
        
        # Scale features
        X_scaled = preprocessor.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=preprocessor.feature_columns)
        
        # Get model
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available. Please train the model first.'}), 404
        
        model = models[model_name]
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Decode labels
        if hasattr(preprocessor.label_encoder, 'classes_'):
            predicted_labels = preprocessor.label_encoder.inverse_transform(predictions)
        else:
            predicted_labels = predictions
        
        # Count predictions
        unique, counts = np.unique(predicted_labels, return_counts=True)
        prediction_counts = dict(zip(unique, counts))
        
        # Calculate accuracy if labels are available
        accuracy = None
        if 'label' in df.columns or 'attack' in df.columns or 'class' in df.columns:
            target_col = 'label' if 'label' in df.columns else ('attack' if 'attack' in df.columns else 'class')
            y_true = df[target_col].values
            if hasattr(preprocessor.label_encoder, 'classes_'):
                y_true_encoded = preprocessor.label_encoder.transform(y_true)
            else:
                y_true_encoded = y_true
            accuracy = float(np.mean(predictions == y_true_encoded) * 100)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'total_samples': len(predictions),
            'predictions': prediction_counts,
            'accuracy': accuracy,
            'model': model_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models."""
    available_models = list(models.keys())
    return jsonify({
        'available_models': available_models,
        'total': len(available_models)
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    stats = {
        'models_loaded': len(models),
        'available_models': list(models.keys()),
        'preprocessor_loaded': preprocessor is not None
    }
    return jsonify(stats)


@app.route('/api/train', methods=['POST'])
def train_model():
    """API endpoint for training models (admin only)."""
    try:
        # This would require authentication in production
        data = request.json
        model_type = data.get('model_type', 'random_forest')
        data_path = data.get('data_path')
        
        if not data_path or not os.path.exists(data_path):
            return jsonify({'error': 'Invalid data path'}), 400
        
        # Load and preprocess data
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(data_path)
        df_cleaned = preprocessor.clean_data(df)
        df_features = preprocessor.extract_features(df_cleaned)
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(df_features)
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestIDS()
        elif model_type == 'svm':
            model = SVMIDS()
        elif model_type == 'neural_network':
            model = NeuralNetworkIDS()
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        model.train(X_train, y_train)
        
        # Save model
        if model_type == 'neural_network':
            model.save(f'models/{model_type}_model.h5')
        else:
            model.save(f'models/{model_type}_model.pkl')
        
        preprocessor.save_preprocessor('models/preprocessor.pkl')
        
        # Reload models
        load_models()
        
        return jsonify({'success': True, 'message': f'{model_type} model trained successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting IDS ML Web Application...")
    print("Available models:", list(models.keys()))
    print("Preprocessor loaded:", preprocessor is not None)
    print("\nAccess the web interface at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

