"""
Feature engineering for population health data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Feature engineering for population health analysis."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""
        logger.info("Creating engineered features...")
        
        data = data.copy()
        
        # Age-based features
        data['age_squared'] = data['age'] ** 2
        data['age_group'] = pd.cut(data['age'], bins=[0, 30, 50, 70, 100], 
                                  labels=['18-30', '31-50', '51-70', '>70'])
        
        # Risk interaction features
        data['age_risk_interaction'] = data['age'] * data['composite_risk_score']
        data['chronic_risk_interaction'] = data['chronic_conditions'] * data['composite_risk_score']
        
        # Utilization features
        data['total_prior_utilization'] = data['prior_ed_visits'] + data['prior_hospitalizations']
        data['utilization_intensity'] = data['total_prior_utilization'] / (data['age'] + 1)
        
        # Binary feature combinations
        if 'substance_use_disorder' in data.columns and 'mental_health_condition' in data.columns:
            data['dual_diagnosis'] = data['substance_use_disorder'] & data['mental_health_condition']
        
        logger.info(f"Created {data.shape[1]} total features")
        return data
