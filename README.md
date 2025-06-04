# 🛡️ Fraud Detection System

A comprehensive machine learning system for detecting fraudulent transactions using advanced classification algorithms and anomaly detection techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This fraud detection system is designed to identify potentially fraudulent transactions using machine learning techniques. The project implements various classification algorithms and anomaly detection methods to achieve high accuracy while minimizing false positives.

## ✨ Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Multiple ML Models**: Implementation of various classification algorithms
- **Anomaly Detection**: Advanced techniques for identifying unusual patterns
- **Model Evaluation**: Detailed performance metrics and visualization
- **Real-time Prediction**: Efficient prediction pipeline for new transactions
- **Interactive Notebooks**: Jupyter notebooks for data exploration and analysis

## 📁 Project Structure

```
fraud_detection_project/
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── model_training.py         # ML model training scripts
│   ├── evaluate.py              # Model evaluation and metrics
│   ├── anomaly_detection.py     # Anomaly detection algorithms
│   └── utils.py                 # Utility functions
├── notebooks/                   # Jupyter notebooks
│   └── data_exploration.ipynb   # Data exploration and analysis
├── data/                        # Data directory (ignored in git)
│   ├── raw/                     # Raw datasets
│   └── processed/               # Processed datasets
├── models/                      # Trained model files
├── env/                         # Virtual environment (ignored in git)
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
└── fraud_detection.log          # Application logs
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/shreyashsing/fraud_detection.git
   cd fraud_detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv env
   
   # On Windows
   env\Scripts\activate
   
   # On macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd fraud_detection_project
   pip install -r requirements.txt
   ```

## 💻 Usage

### Quick Start

1. **Run the main script**
   ```bash
   python main.py
   ```

2. **Explore data with Jupyter notebooks**
   ```bash
   jupyter notebook notebooks/data_exploration.ipynb
   ```

### Training Models

```python
from src.model_training import train_model
from src.data_preprocessing import preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocess_data('data/raw/transactions.csv')

# Train model
model = train_model(X_train, y_train)
```

### Making Predictions

```python
from src.evaluate import predict_fraud

# Predict on new data
predictions = predict_fraud(model, new_transaction_data)
```

## 📊 Data

The system works with transaction datasets containing features such as:

- Transaction amount
- Merchant information
- User behavior patterns
- Temporal features
- Geographic data

**Note**: Due to privacy and size constraints, actual datasets are not included in the repository. Please ensure your data follows the expected format described in the preprocessing module.

## 🤖 Models

The system implements several machine learning algorithms:

- **Random Forest**: Ensemble method for robust classification
- **Gradient Boosting**: Advanced boosting technique for high accuracy
- **Neural Networks**: Deep learning for complex pattern recognition
- **Isolation Forest**: Anomaly detection for outlier identification
- **SVM**: Support Vector Machines for classification

## 📈 Results

The models are evaluated using various metrics:

- **Precision**: Minimizing false positives
- **Recall**: Capturing actual fraud cases
- **F1-Score**: Balanced performance measure
- **AUC-ROC**: Overall model performance
- **Confusion Matrix**: Detailed classification results

## 🛠️ Dependencies

```
pandas==2.0.3
numpy==1.26.0
scikit-learn==1.4.2
tensorflow==2.15.0
imbalanced-learn==0.12.0
matplotlib==3.8.0
seaborn==0.13.0
joblib==1.4.2
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Shreyash Singh**
- GitHub: [@shreyashsing](https://github.com/shreyashsing)

## 🔗 Links

- [Project Repository](https://github.com/shreyashsing/fraud_detection)
- [Issues](https://github.com/shreyashsing/fraud_detection/issues)
- [Documentation](https://github.com/shreyashsing/fraud_detection/wiki)

---

⭐ **Star this repository if you find it helpful!** 