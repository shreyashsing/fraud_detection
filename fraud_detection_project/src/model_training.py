import numpy as np  # Add this import
import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_dim):
    # Encoder
    encoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu')  # Bottleneck
    ])
    
    # Decoder
    decoder = models.Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    
    # Autoencoder
    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def build_classifier(encoder, input_dim):
    model = models.Sequential([
        encoder,
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_models(X, y):
    input_dim = X.shape[1]
    
    # Train autoencoder on normal transactions (non-fraud)
    autoencoder, encoder = build_autoencoder(input_dim)
    X_normal = X[y == 0]
    autoencoder.fit(X_normal, X_normal, epochs=20, batch_size=32, validation_split=0.2)
    
    # Train classifier on balanced data
    classifier = build_classifier(encoder, input_dim)
    classifier.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save models
    autoencoder.save('../models/final/autoencoder.h5')
    classifier.save('../models/final/classifier.h5')
    
    return autoencoder, classifier

if __name__ == "__main__":
    X = np.load('../data/processed/X_balanced.npy')  # np is now defined
    y = np.load('../data/processed/y_balanced.npy')  # np is now defined
    autoencoder, classifier = train_models(X, y)