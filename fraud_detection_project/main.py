import logging
import numpy as np  # Add this import for potential future use or consistency
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_models
from src.anomaly_detection import detect_anomalies
from src.evaluate import evaluate_model

# Configure logging
logging.basicConfig(filename='fraud_detection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting fraud detection pipeline")
    
    # Preprocess data
    X, y, scaler = load_and_preprocess_data('data/raw/creditcard.csv')
    logging.info("Data preprocessing completed")
    
    # Train models
    autoencoder, classifier = train_models(X, y)
    logging.info("Model training completed")
    
    # Detect anomalies
    anomalies, mse = detect_anomalies(autoencoder, X)
    logging.info(f"Detected {np.sum(anomalies)} anomalies")  # np used here
    
    # Evaluate classifier
    accuracy = evaluate_model(classifier, X, y)
    logging.info(f"Model accuracy: {accuracy:.4f}")
    
    if accuracy >= 0.95:
        logging.info("Model achieved 95%+ accuracy!")
    else:
        logging.warning("Model accuracy below 95%. Consider tuning hyperparameters.")

if __name__ == "__main__":
    main()