"""
Traditional machine learning models for population health analysis.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import logging

logger = logging.getLogger(__name__)

class TraditionalModels:
    """Traditional statistical and ML models."""
    
    def __init__(self):
        self.models = {}
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train logistic regression models."""
        models = {}
        
        for horizon in ['7d', '30d', '90d', '180d']:
            if horizon in y_train:
                logger.info(f"Training logistic regression for {horizon}")
                
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['liblinear']
                }
                
                lr = LogisticRegression(random_state=42, max_iter=1000)
                grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc')
                grid_search.fit(X_train, y_train[horizon])
                
                models[f'logistic_regression_{horizon}'] = grid_search.best_estimator_
        
        return models
    
    def train_cox_model(self, X_train, y_train, X_val, y_val):
        """Train Cox proportional hazards model."""
        if 'time_to_event' not in y_train:
            return {}
        
        logger.info("Training Cox proportional hazards model")
        
        # Prepare data for Cox model
        import pandas as pd
        df_train = pd.DataFrame(X_train)
        df_train['T'] = y_train['time_to_event']
        df_train['E'] = y_train['7d']  # Event indicator
        
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df_train, duration_col='T', event_col='E')
        
        return {'cox_model': cph}
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all traditional models."""
        models = {}
        
        # Logistic regression
        models.update(self.train_logistic_regression(X_train, y_train, X_val, y_val))
        
        # Cox model
        models.update(self.train_cox_model(X_train, y_train, X_val, y_val))
        
        return models
