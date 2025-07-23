"""
Configuration settings for the population health ML analysis.
"""

import os
from typing import Dict, Any, List
import yaml

# Default configuration
DEFAULT_CONFIG = {
    # Data settings
    'data': {
        'random_state': 42,
        'test_size': 0.15,
        'validation_size': 0.15,
        'temporal_split': True,
        'train_end_date': '2022-06-30',
        'validation_end_date': '2022-12-31',
        'test_start_date': '2023-01-01'
    },
    
    # Model settings
    'models': {
        'time_horizons': [7, 30, 90, 180],
        'cv_folds': 5,
        'n_bootstrap': 1000,
        'confidence_level': 0.95,
        'hyperparameter_trials': 100
    },
    
    # Traditional models
    'traditional': {
        'logistic_regression': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': 1000
        },
        'cox_model': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': 1000
        }
    },
    
    # Ensemble models
    'ensemble': {
        'random_forest': {
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'random_survival_forest': {
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 10, 15],
            'min_samples_split': [6, 10, 20],
            'min_samples_leaf': [3, 6, 10]
        },
        'xgboost': {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    },
    
    # Deep learning models
    'deep_learning': {
        'deepsurv': {
            'hidden_layers': [[64, 32, 16], [128, 64, 32], [256, 128, 64]],
            'dropout': [0.1, 0.3, 0.5],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [64, 128, 256],
            'epochs': 100,
            'patience': 10
        },
        'deephit': {
            'hidden_layers': [[64, 32, 16], [128, 64, 32]],
            'dropout': [0.1, 0.3, 0.5],
            'learning_rate': [0.001, 0.01],
            'batch_size': [64, 128, 256],
            'epochs': 100,
            'patience': 10
        }
    },
    
    # Causal inference models
    'causal_inference': {
        't_learner': {
            'base_model': 'random_forest',
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15]
        },
        'x_learner': {
            'base_model': 'random_forest',
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15]
        },
        'causal_forest': {
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 10, 15],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10]
        }
    },
    
    # Reinforcement learning models
    'reinforcement_learning': {
        'contextual_bandits': {
            'alpha': [0.1, 0.5, 1.0, 2.0],
            'exploration_strategy': ['ucb', 'epsilon_greedy'],
            'epsilon': [0.1, 0.2, 0.3]
        },
        'dqn': {
            'hidden_layers': [[64, 32], [128, 64]],
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64],
            'memory_size': 10000,
            'epsilon_decay': 0.995
        }
    },
    
    # Evaluation settings
    'evaluation': {
        'metrics': [
            'auc', 'sensitivity', 'specificity', 'ppv', 'npv',
            'nnt', 'brier_score', 'calibration_slope', 'calibration_intercept'
        ],
        'clinical_validation': {
            'n_cases': 200,
            'n_reviewers': 3,
            'intervention_categories': ['high_impact', 'medium_impact', 'low_impact']
        }
    },
    
    # Visualization settings
    'visualization': {
        'figure_size': (12, 8),
        'dpi': 300,
        'format': 'png',
        'color_palette': 'Set2',
        'font_size': 12
    },
    
    # Output settings
    'output': {
        'results_dir': 'results/',
        'models_dir': 'results/models/',
        'figures_dir': 'results/figures/',
        'tables_dir': 'results/tables/',
        'validation_dir': 'results/validation/'
    }
}


class Config:
    """Configuration manager for the analysis."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Recursively update configuration
        self._update_config(self.config, user_config)
    
    def _update_config(self, base_config: Dict[str, Any], 
                      user_config: Dict[str, Any]):
        """Recursively update configuration.
        
        Args:
            base_config: Base configuration dictionary
            user_config: User configuration dictionary
        """
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save_config(self, config_path: str):
        """Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file
        """
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def get_model_params(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter grid for a specific model.
        
        Args:
            model_type: Type of model (e.g., 'traditional', 'ensemble')
            model_name: Name of the model
            
        Returns:
            Hyperparameter grid
        """
        return self.get(f'{model_type}.{model_name}', {})
    
    def get_time_horizons(self) -> List[int]:
        """Get time horizons for prediction.
        
        Returns:
            List of time horizons in days
        """
        return self.get('models.time_horizons', [7, 30, 90, 180])
    
    def get_random_state(self) -> int:
        """Get random state for reproducibility.
        
        Returns:
            Random state integer
        """
        return self.get('data.random_state', 42)


# Global configuration instance
config = Config()

