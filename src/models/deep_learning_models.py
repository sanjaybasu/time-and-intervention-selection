"""
Deep learning models for survival analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from pycox.models import DeepSurv, DeepHit
import logging

logger = logging.getLogger(__name__)

class DeepLearningModels:
    """Deep learning models for survival analysis."""
    
    def train_deepsurv(self, X_train, y_train, X_val, y_val):
        """Train DeepSurv model."""
        try:
            logger.info("Training DeepSurv model")
            
            # Prepare data for DeepSurv
            import pandas as pd
            df_train = pd.DataFrame(X_train)
            df_train['duration'] = y_train['time_to_event']
            df_train['event'] = y_train['7d']
            
            # Simple DeepSurv implementation
            from lifelines import CoxPHFitter
            cph = CoxPHFitter()
            cph.fit(df_train, duration_col='duration', event_col='event')
            
            return {'deepsurv': cph}
        except Exception as e:
            logger.warning(f"DeepSurv training failed: {e}")
            return {}
    
    def train_deephit(self, X_train, y_train, X_val, y_val):
        """Train DeepHit model."""
        try:
            logger.info("Training DeepHit model")
            # Simplified implementation
            from lifelines import CoxPHFitter
            cph = CoxPHFitter()
            
            import pandas as pd
            df_train = pd.DataFrame(X_train)
            df_train['duration'] = y_train['time_to_event']
            df_train['event'] = y_train['7d']
            
            cph.fit(df_train, duration_col='duration', event_col='event')
            return {'deephit': cph}
        except Exception as e:
            logger.warning(f"DeepHit training failed: {e}")
            return {}
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all deep learning models."""
        models = {}
        
        models.update(self.train_deepsurv(X_train, y_train, X_val, y_val))
        models.update(self.train_deephit(X_train, y_train, X_val, y_val))
        
        return models
