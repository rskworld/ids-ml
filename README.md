# Intrusion Detection System (IDS) with Machine Learning

A machine learning-based intrusion detection system to identify network attacks and anomalies in real-time.

**Version**: 1.0.0  
**Repository**: [https://github.com/rskworld/ids-ml](https://github.com/rskworld/ids-ml)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Web Interface](#web-interface)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Dataset Requirements](#dataset-requirements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

This project implements an Intrusion Detection System using machine learning algorithms to detect malicious network activities. It analyzes network traffic patterns, packet headers, and flow data to identify various attack types including DoS, DDoS, port scans, and unauthorized access attempts.

## Features

- **Network traffic analysis and feature extraction**: Comprehensive feature engineering from network packet data
- **Multiple ML algorithms**: Random Forest, SVM, and Neural Networks for attack detection
- **Real-time attack detection**: Live monitoring and classification of network traffic
- **Performance metrics and confusion matrix**: Detailed evaluation of model performance
- **Visualization of attack patterns**: Interactive plots and charts for data analysis
- **Web Interface**: Complete web application with demo, documentation, and contact pages
- **REST API**: Programmatic access to ML models via API endpoints

## Technologies

- Python 3.8+
- Scikit-learn
- TensorFlow
- Pandas
- NumPy
- Jupyter Notebook
- Flask (for web interface)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

### Installation Steps

#### 1. Clone or Download the Project

```bash
git clone https://github.com/rskworld/ids-ml.git
cd ids-ml
```

Or extract the downloaded ZIP file and navigate to the project directory.

#### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with TensorFlow installation, you may need to:
- Use a specific TensorFlow version compatible with your system
- Install CPU-only version: `pip install tensorflow-cpu==2.15.0`

#### 4. Verify Installation

Run the example script to verify everything is set up correctly:
```bash
python example_usage.py
```

This will:
- Create sample data
- Preprocess the data
- Train a simple model
- Evaluate and save the model

## Quick Start

### Option 1: Using Command Line Scripts

1. **Train all models:**
   ```bash
   python scripts/train.py --data data/raw/your_dataset.csv
   ```

2. **Make predictions:**
   ```bash
   python scripts/predict.py --input data/raw/test_data.csv --model random_forest
   ```

### Option 2: Using Jupyter Notebooks

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open and run notebooks in order:**
   - `notebooks/data_exploration.ipynb` - Explore your dataset
   - `notebooks/model_training.ipynb` - Train models
   - `notebooks/evaluation.ipynb` - Evaluate models

### Option 3: Using Python API

```python
from src.preprocessing import DataPreprocessor
from src.models import RandomForestIDS
from src.evaluation import ModelEvaluator

# Load and preprocess data
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data/raw/your_data.csv')
df_cleaned = preprocessor.clean_data(df)
df_features = preprocessor.extract_features(df_cleaned)
X_train, X_test, y_train, y_test = preprocessor.prepare_data(df_features)

# Train model
model = RandomForestIDS()
model.train(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator("Random Forest")
y_pred = model.predict(X_test)
metrics = evaluator.evaluate(y_test, y_pred)
```

## Usage

### Web Interface

Start the web application:
```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

The web interface includes:
- **Home Page**: Project overview and features
- **Demo Page**: Upload CSV files and get real-time predictions
- **Documentation**: Complete usage guide
- **Contact Page**: Get in touch with the team
- **About Page**: Learn more about the project

### Data Preparation

1. Place your network traffic dataset in the `data/raw/` directory
2. Run the preprocessing script:
```bash
python src/preprocessing.py
```

### Model Training

Train all models:
```bash
python scripts/train.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/model_training.ipynb
```

### Real-time Detection

Run the real-time detection script:
```bash
python scripts/predict.py --input <network_traffic_file>
```

Or use the web interface to upload files and get predictions through the browser.

### Jupyter Notebooks

Explore the data and models using the provided notebooks:
- `notebooks/data_exploration.ipynb`: Data analysis and visualization
- `notebooks/model_training.ipynb`: Model training and comparison
- `notebooks/evaluation.ipynb`: Model evaluation and metrics

## Web Interface

### Running the Web Application

#### 1. Train Models (First Time)

Before using the web interface, you need to train the models:

```bash
python scripts/train.py
```

This will create the necessary model files in the `models/` directory.

#### 2. Start the Web Server

```bash
python app.py
```

The application will start on `http://localhost:5000`

#### 3. Access the Web Interface

Open your browser and navigate to:
- Home: `http://localhost:5000/`
- Demo: `http://localhost:5000/demo.html`
- Contact: `http://localhost:5000/contact.html`
- Documentation: `http://localhost:5000/documentation.html`

### Web Pages

1. **Home Page** (`index.html`)
   - Project overview and hero section
   - Key features showcase
   - Technologies used
   - Statistics and metrics

2. **About Page** (`about.html`)
   - Project overview
   - Team information (Molla Samser & Rima Khatun)
   - Technology stack details

3. **Demo Page** (`demo.html`)
   - Upload CSV files for analysis
   - Real-time attack detection
   - Model selection (Random Forest, SVM, Neural Network)
   - Results visualization

4. **Documentation Page** (`documentation.html`)
   - Installation instructions
   - Usage examples
   - API reference
   - Dataset format guide

5. **Contact Page** (`contact.html`)
   - Contact information
   - General inquiry form
   - Content removal request form
   - Statistics display

6. **Legal Pages**
   - Privacy Policy (`privacy.html`)
   - Terms & Conditions (`terms.html`)
   - Disclaimer (`disclaimer.html`)

### Web Interface Structure

```
web/
├── index.html          # Home page
├── about.html          # About page
├── demo.html           # Demo/upload page
├── documentation.html  # Documentation
├── contact.html        # Contact page
├── privacy.html        # Privacy policy
├── terms.html          # Terms & conditions
├── disclaimer.html     # Disclaimer
├── css/
│   └── style.css       # Main stylesheet
└── js/
    ├── main.js         # Main JavaScript
    ├── contact.js      # Contact form handling
    └── demo.js         # Demo page functionality
```

## API Endpoints

### POST /api/predict
Upload a CSV file and get predictions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameters:
  - `file`: CSV file (required)
  - `model`: Model name (random_forest, svm, neural_network)

**Response:**
```json
{
  "success": true,
  "total_samples": 100,
  "predictions": {
    "normal": 80,
    "dos": 15,
    "probe": 5
  },
  "accuracy": 95.5,
  "model": "random_forest"
}
```

### GET /api/models
List available trained models.

**Response:**
```json
{
  "available_models": ["random_forest", "svm", "neural_network"],
  "total": 3
}
```

### GET /api/stats
Get system statistics.

**Response:**
```json
{
  "models_loaded": 3,
  "available_models": ["random_forest", "svm", "neural_network"],
  "preprocessor_loaded": true
}
```

### POST /api/train
Train a new model (requires authentication in production).

## Project Structure

```
ids-ml/
├── data/                    # Dataset directory
│   ├── raw/                # Raw network traffic data
│   └── processed/          # Processed features
├── models/                  # Trained model files
├── notebooks/               # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── src/                     # Source code
│   ├── preprocessing.py     # Data preprocessing
│   ├── models.py           # ML model definitions
│   ├── evaluation.py       # Model evaluation
│   ├── visualization.py    # Visualization utilities
│   └── feature_extraction.py # Feature engineering
├── scripts/                 # Utility scripts
│   ├── train.py            # Training script
│   └── predict.py          # Prediction script
├── web/                     # Web interface
│   ├── index.html
│   ├── css/
│   └── js/
├── uploads/                 # Upload directory
├── app.py                   # Flask web application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Dataset Requirements

The project expects network traffic data with the following structure:

### Required Columns:
- At least one numeric feature column
- A target column named one of: `label`, `attack`, `class`, or `target`

### Supported Datasets:
- NSL-KDD
- UNSW-NB15
- CICIDS2017
- Custom network traffic data

### Sample Data Format:
```csv
duration,protocol_type,service,src_bytes,dst_bytes,count,label
0,tcp,http,1000,2000,5,normal
1,udp,ftp,500,1500,3,dos
...
```

## Model Performance

The system includes three ML algorithms:
- **Random Forest**: Fast training, good interpretability
- **SVM**: Effective for high-dimensional data
- **Neural Network**: Deep learning approach for complex patterns

## Troubleshooting

### Issue: Import errors
**Solution:** Make sure you've activated the virtual environment and installed all dependencies.

### Issue: TensorFlow installation fails
**Solution:** 
- Try: `pip install tensorflow-cpu==2.15.0`
- Or use a different TensorFlow version compatible with your Python version

### Issue: Memory errors during training
**Solution:**
- Reduce dataset size
- Use smaller models (fewer estimators for Random Forest)
- Process data in batches

### Issue: Model training is slow
**Solution:**
- Use smaller datasets for testing
- Reduce model complexity
- Use CPU-optimized versions of libraries

### Models Not Loading
- Ensure models are trained: `python scripts/train.py`
- Check that model files exist in `models/` directory
- Verify preprocessor is saved: `models/preprocessor.pkl`

### File Upload Issues
- Check file size (max 10MB)
- Ensure file is CSV format
- Verify file has correct column structure

### API Errors
- Check Flask console for error messages
- Verify models are loaded on startup
- Ensure data format matches expected structure

## Security Notes

For production deployment:
1. Add authentication to `/api/train` endpoint
2. Implement rate limiting
3. Add CSRF protection
4. Use HTTPS
5. Validate and sanitize all inputs
6. Implement file size limits
7. Add logging and monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

Copyright (c) 2025 Molla Samser & Rima Khatun (RSK World)  
Copyright (c) 2025 rskworld.in

## Author

**Molla Samser** - Founder  
**Rima Khatun** - Designer & Tester

### Contact Information

- **Email**: help@rskworld.in | support@rskworld.in
- **Phone**: +91 93305 39277
- **Website**: [rskworld.in](https://rskworld.in)
- **Location**: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147

### About RSK World

Founded by Molla Samser, with Designer & Tester Rima Khatun, RSK World is your one-stop destination for free programming resources, source code, and development tools.

---

**Version**: 1.0.0  
**Repository**: [https://github.com/rskworld/ids-ml](https://github.com/rskworld/ids-ml)
