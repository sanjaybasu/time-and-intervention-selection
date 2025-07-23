"""
Evaluation metrics for population health models.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class Metrics:
    """Evaluation metrics for ML models."""
    
    def calculate_auc(self, y_true, y_pred_proba):
        """Calculate AUC with confidence interval."""
        auc = roc_auc_score(y_true, y_pred_proba)
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_aucs = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[indices])) > 1:
                bootstrap_auc = roc_auc_score(y_true[indices], y_pred_proba[indices])
                bootstrap_aucs.append(bootstrap_auc)
        
        ci_lower = np.percentile(bootstrap_aucs, 2.5)
        ci_upper = np.percentile(bootstrap_aucs, 97.5)
        
        return auc, (ci_lower, ci_upper)
    
    def calculate_nnt(self, y_true, y_pred_proba, threshold=0.5):
        """Calculate Number Needed to Treat."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # True positives and false positives
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        if tp == 0:
            return np.inf, (np.inf, np.inf)
        
        # NNT = 1 / (TP rate - FP rate)
        tp_rate = tp / np.sum(y_true == 1)
        fp_rate = fp / np.sum(y_true == 0)
        
        if tp_rate <= fp_rate:
            return np.inf, (np.inf, np.inf)
        
        nnt = 1 / (tp_rate - fp_rate)
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_nnts = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            tp_boot = np.sum((y_pred_boot == 1) & (y_true_boot == 1))
            fp_boot = np.sum((y_pred_boot == 1) & (y_true_boot == 0))
            
            if tp_boot > 0:
                tp_rate_boot = tp_boot / np.sum(y_true_boot == 1)
                fp_rate_boot = fp_boot / np.sum(y_true_boot == 0)
                
                if tp_rate_boot > fp_rate_boot:
                    nnt_boot = 1 / (tp_rate_boot - fp_rate_boot)
                    if nnt_boot < 1000:  # Cap extreme values
                        bootstrap_nnts.append(nnt_boot)
        
        if len(bootstrap_nnts) > 0:
            ci_lower = np.percentile(bootstrap_nnts, 2.5)
            ci_upper = np.percentile(bootstrap_nnts, 97.5)
        else:
            ci_lower, ci_upper = np.inf, np.inf
        
        return nnt, (ci_lower, ci_upper)
    
    def evaluate_all_models(self, models, X_test, y_test):
        """Evaluate all models."""
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}")
            
            model_results = {}
            
            # Get predictions for each time horizon
            for horizon in ['7d', '30d', '90d', '180d']:
                if horizon in y_test:
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                        elif hasattr(model, 'predict'):
                            y_pred_proba = model.predict(X_test)
                        else:
                            continue
                        
                        # Calculate metrics
                        auc, auc_ci = self.calculate_auc(y_test[horizon], y_pred_proba)
                        nnt, nnt_ci = self.calculate_nnt(y_test[horizon], y_pred_proba)
                        
                        model_results[horizon] = {
                            'auc': auc,
                            'auc_ci': auc_ci,
                            'nnt': nnt,
                            'nnt_ci': nnt_ci
                        }
                    except Exception as e:
                        logger.warning(f"Evaluation failed for {model_name} {horizon}: {e}")
            
            results[model_name] = model_results
        
        return results
    
    def calculate_clinical_utility(self, models, X_test, y_test, results):
        """Calculate clinical utility metrics."""
        # Add clinical utility calculations
        for model_name in results:
            if '7d' in results[model_name]:
                results[model_name]['clinical_utility'] = {
                    'high_impact': True if results[model_name]['7d']['nnt'] < 5 else False,
                    'resource_efficient': True if results[model_name]['7d']['auc'] > 0.75 else False
                }
        
        return results
