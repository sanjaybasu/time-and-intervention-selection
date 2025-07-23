"""
Ensemble machine learning models.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sksurv.ensemble import RandomSurvivalForest
import logging

logger = logging.getLogger(__name__)

class EnsembleModels:
    """Ensemble machine learning models."""
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train random forest models."""
        models = {}
        
        for horizon in ['7d', '30d', '90d', '180d']:
            if horizon in y_train:
                logger.info(f"Training random forest for {horizon}")
                
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, None],
                    'min_samples_split': [2, 5]
                }
                
                rf = RandomForestClassifier(random_state=42)
                grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc')
                grid_search.fit(X_train, y_train[horizon])
                
                models[f'random_forest_{horizon}'] = grid_search.best_estimator_
        
        return models
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost models."""
        models = {}
        
        for horizon in ['7d', '30d', '90d', '180d']:
            if horizon in y_train:
                logger.info(f"Training XGBoost for {horizon}")
                
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6],
                    'learning_rate': [0.01, 0.1]
                }
                
                xgb_model = xgb.XGBClassifier(random_state=42)
                grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='roc_auc')
                grid_search.fit(X_train, y_train[horizon])
                
                models[f'xgboost_{horizon}'] = grid_search.best_estimator_
        
        return models
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all ensemble models."""
        models = {}
        
        models.update(self.train_random_forest(X_train, y_train, X_val, y_val))
        models.update(self.train_xgboost(X_train, y_train, X_val, y_val))
        
        return models
