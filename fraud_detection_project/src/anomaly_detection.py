import numpy as np
from tensorflow.keras.models import load_model

def detect_anomalies(autoencoder, X, threshold=95):
    # Calculate reconstruction error
    X_reconstructed = autoencoder.predict(X)
    mse = np.mean(np.square(X - X_reconstructed), axis=1)
    
    # Set threshold based on percentile
    threshold_value = np.percentile(mse, threshold)
    anomalies = mse > threshold_value
    return anomalies, mse

if __name__ == "__main__":
    autoencoder = load_model('../models/final/autoencoder.h5')
    X = np.load('../data/processed/X_balanced.npy')
    anomalies, mse = detect_anomalies(autoencoder, X)
    print(f"Detected {np.sum(anomalies)} anomalies")