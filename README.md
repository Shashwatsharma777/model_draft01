# Fuel Event Classification Model

A machine learning system for classifying vehicle fuel events using CatBoost classifier. This project includes model training, evaluation, and a Flask API for real-time predictions.

## 🚀 Features

- **CatBoost Classifier**: High-performance gradient boosting for fuel event classification
- **Flask API**: RESTful API for real-time predictions
- **Batch Processing**: Command-line tools for processing large datasets
- **Model Evaluation**: Comprehensive evaluation metrics and visualization
- **Feature Engineering**: Automatic timestamp and fuel level feature extraction

## 📁 Project Structure

```
SIDEMEN copy/
├── app.py                 # Flask API server
├── config.py             # Configuration and paths
├── model/
│   ├── model.py          # Model training script
│   └── catboost_model_random_tuned.cbm  # Trained model
├── data/                 # Data files and evaluation results
├── predict.py            # Batch prediction script
├── evaluate.py           # Model evaluation script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "SIDEMEN copy"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Model Training

Train a new CatBoost model:

```bash
cd model
python model.py
```

This will:
- Load the training dataset
- Perform feature engineering
- Train the model with hyperparameter tuning
- Save the trained model and artifacts

### 2. Flask API Server

Start the prediction API:

```bash
python app.py
```

The server will run on `http://localhost:5001` by default.

#### API Endpoints

- **Health Check**: `GET /health`
- **Predictions**: `POST /predict`

#### Making Predictions

**JSON Input:**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "fuelLevel": 75.5,
        "previous_fuel_level": 80.0,
        "timestamp": "2024-01-15T10:30:00",
        "isOverSpeed": false,
        "ignitionStatus": "ON"
      }
    ]
  }'
```

**CSV File Upload:**
```bash
curl -X POST http://localhost:5001/predict \
  -F "file=@your_data.csv"
```

**With Probabilities:**
```bash
curl -X POST "http://localhost:5001/predict?proba=true" \
  -H "Content-Type: application/json" \
  -d '{"data": [...]}'
```

### 3. Batch Predictions

Process large datasets using the command-line tool:

```bash
python predict.py --input data/input.csv --output data/predictions.csv --proba
```

Options:
- `--input`: Input CSV file path
- `--output`: Output CSV file path
- `--proba`: Include prediction probabilities
- `--output_proba`: Custom path for probabilities file

### 4. Model Evaluation

Evaluate model performance:

```bash
python evaluate.py --input data/predictions.csv --outdir data/
```

This generates:
- `evaluation_summary.txt`: Overall performance metrics
- `classification_report.csv`: Detailed classification metrics
- `confusion_matrix.csv`: Confusion matrix data

## 📊 Data Format

### Input Features

The model expects the following features:

- **fuelLevel**: Current fuel level (numeric)
- **previous_fuel_level**: Previous fuel level (numeric)
- **timestamp**: Timestamp (datetime string)
- **isOverSpeed**: Over speed flag (boolean/string)
- **ignitionStatus**: Ignition status ("ON"/"OFF")

### Generated Features

The system automatically creates:
- **fuel_diff**: Difference between current and previous fuel levels
- **hour**: Hour of day (0-23)
- **day_of_week**: Day of week (0-6, Monday=0)
- **is_weekend**: Weekend flag (0/1)

### Output

- **predictions**: Predicted event type
- **probabilities**: Class probability scores (optional)

## ⚙️ Configuration

Edit `config.py` to customize:
- Model and artifact paths
- Server host and port
- Default input/output file paths
- Debug mode settings

## 🔧 Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for functions and classes

### Testing
```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
pytest

# Code formatting
black .

# Linting
flake8 .
```

## 📈 Model Performance

The CatBoost model includes:
- **Hyperparameter Tuning**: Randomized search for optimal parameters
- **Class Balancing**: Automatic class weight calculation
- **Cross-validation**: 3-fold CV during training
- **Feature Engineering**: Automatic timestamp and fuel level processing

## 🚨 Troubleshooting

### Common Issues

1. **Model not found**: Ensure the trained model file exists in the model directory
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Port already in use**: Change the port in `config.py`
4. **Data format errors**: Check that input data matches expected schema

### Logs

Check the console output for detailed error messages and model training progress.


