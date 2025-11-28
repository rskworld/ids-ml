# Setup Guide for IDS ML Project

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation Steps

### 1. Clone or Download the Project

If you have the project URL:
```bash
git clone https://github.com/rskworld/ids-ml.git
cd ids-ml
```

Or extract the downloaded ZIP file and navigate to the project directory.

### 2. Create a Virtual Environment (Recommended)

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

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with TensorFlow installation, you may need to:
- Use a specific TensorFlow version compatible with your system
- Install CPU-only version: `pip install tensorflow-cpu==2.15.0`

### 4. Verify Installation

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

## Project Structure

```
ids-ml/
├── data/              # Data directory
│   ├── raw/          # Raw datasets
│   └── processed/    # Processed data
├── models/            # Trained models
├── notebooks/        # Jupyter notebooks
├── scripts/           # Training and prediction scripts
├── src/              # Source code
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── feature_extraction.py
├── requirements.txt   # Python dependencies
├── README.md         # Project documentation
└── SETUP.md          # This file
```

## Next Steps

1. Replace sample data with your own dataset
2. Explore the data using `notebooks/data_exploration.ipynb`
3. Train models using `scripts/train.py` or the training notebook
4. Evaluate model performance
5. Use trained models for real-time detection

## Support

For issues and questions, please refer to the main README.md or create an issue on the project repository.

