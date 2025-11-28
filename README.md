# Intrusion Detection System (IDS) with Machine Learning

A machine learning-based intrusion detection system to identify network attacks and anomalies in real-time.

## Overview

This project implements an Intrusion Detection System using machine learning algorithms to detect malicious network activities. It analyzes network traffic patterns, packet headers, and flow data to identify various attack types including DoS, DDoS, port scans, and unauthorized access attempts.

## Features

- **Network traffic analysis and feature extraction**: Comprehensive feature engineering from network packet data
- **Multiple ML algorithms**: Random Forest, SVM, and Neural Networks for attack detection
- **Real-time attack detection**: Live monitoring and classification of network traffic
- **Performance metrics and confusion matrix**: Detailed evaluation of model performance
- **Visualization of attack patterns**: Interactive plots and charts for data analysis

## Technologies

- Python 3.8+
- Scikit-learn
- TensorFlow
- Pandas
- NumPy
- Jupyter Notebook

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
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rskworld/ids-ml.git
cd ids-ml
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
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

### API Endpoints

The Flask application provides REST API endpoints:

- `POST /api/predict` - Upload CSV file and get predictions
- `GET /api/models` - List available trained models
- `GET /api/stats` - Get system statistics
- `POST /api/train` - Train a new model (requires authentication in production)

## Model Performance

The system includes three ML algorithms:
- **Random Forest**: Fast training, good interpretability
- **SVM**: Effective for high-dimensional data
- **Neural Network**: Deep learning approach for complex patterns

## Dataset

This project is designed to work with network intrusion detection datasets such as:
- NSL-KDD
- UNSW-NB15
- CICIDS2017
- Custom network traffic captures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

rskworld.in

