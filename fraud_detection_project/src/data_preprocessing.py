import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Rename columns to match expected names (if using Kaggle dataset)
    df = df.rename(columns={'Time': 'time', 'Amount': 'amount', 'Class': 'is_fraud'})
    
    # Basic cleaning
    df.dropna(inplace=True)
    
    # Feature engineering
    df['hour'] = (df['time'] // 3600) % 24  # Convert seconds to hour of day
    df['amount_log'] = np.log1p(df['amount'])  # Log-transform amount
    
    # Separate features and labels
    X = df.drop(columns=['is_fraud', 'time'])  # Drop 'time' since we derived 'hour'
    y = df['is_fraud']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    
    # Save processed data
    np.save('./data/processed/X_balanced.npy', X_balanced)
    np.save('./data/processed/y_balanced.npy', y_balanced)
    
    return X_balanced, y_balanced, scaler

if __name__ == "__main__":
    X, y, scaler = load_and_preprocess_data('./data/raw/creditcard.csv')