"""
Data preprocessing for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class Preprocessing:
    """Data preprocessing for ML models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def prepare_data(self, data: pd.DataFrame) -> Dict:
        """Prepare data for machine learning."""
        logger.info("Preparing data for ML...")
        
        # Separate features and targets
        feature_cols = [col for col in data.columns if not col.startswith('outcome_') 
                       and col not in ['patient_id', 'time_to_event']]
        
        X = data[feature_cols].copy()
        y = {}
        
        # Create targets for different time horizons
        for horizon in [7, 30, 90, 180]:
            if f'outcome_{horizon}d' in data.columns:
                y[f'{horizon}d'] = data[f'outcome_{horizon}d'].values
        
        # Add time-to-event
        if 'time_to_event' in data.columns:
            y['time_to_event'] = data['time_to_event'].values
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Split data temporally if possible
        if 'data_generation_date' in data.columns:
            # Temporal split
            train_idx = data['data_generation_date'] <= '2022-06-30'
            val_idx = (data['data_generation_date'] > '2022-06-30') & (data['data_generation_date'] <= '2022-12-31')
            test_idx = data['data_generation_date'] > '2022-12-31'
            
            X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
            y_train = {k: v[train_idx] for k, v in y.items()}
            y_val = {k: v[val_idx] for k, v in y.items()}
            y_test = {k: v[test_idx] for k, v in y.items()}
        else:
            # Random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y['7d']
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp['7d']
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_names': X.columns.tolist()
        }
