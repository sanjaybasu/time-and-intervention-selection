"""
Tests for data processing modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_engineering import FeatureEngineering

class TestDataLoader:
    """Test data loading functionality."""
    
    def test_data_validation(self):
        """Test data validation."""
        loader = DataLoader()
        
        # Create test data with required columns
        data = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [25, 45, 65],
            'gender': ['F', 'M', 'F'],
            'race_ethnicity': ['White', 'Black', 'Hispanic'],
            'chronic_conditions': [1, 3, 5],
            'prior_ed_visits': [0, 2, 4],
            'prior_hospitalizations': [0, 1, 2],
            'composite_risk_score': [0.1, 0.5, 0.8],
            'outcome_7d': [0, 1, 1],
            'outcome_30d': [0, 1, 1],
            'outcome_90d': [0, 1, 1],
            'outcome_180d': [0, 1, 1],
            'time_to_event': [180, 15, 5]
        })
        
        # Should not raise an exception
        loader._validate_data(data)
    
    def test_missing_columns(self):
        """Test handling of missing required columns."""
        loader = DataLoader()
        
        # Create data missing required columns
        data = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [25, 45, 65]
        })
        
        with pytest.raises(ValueError):
            loader._validate_data(data)

class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def test_feature_creation(self):
        """Test feature creation."""
        fe = FeatureEngineering()
        
        # Create test data
        data = pd.DataFrame({
            'age': [25, 45, 65],
            'composite_risk_score': [0.1, 0.5, 0.8],
            'chronic_conditions': [1, 3, 5],
            'prior_ed_visits': [0, 2, 4],
            'prior_hospitalizations': [0, 1, 2],
            'substance_use_disorder': [0, 1, 0],
            'mental_health_condition': [1, 1, 0]
        })
        
        result = fe.create_features(data)
        
        # Check that new features were created
        assert 'age_squared' in result.columns
        assert 'age_group' in result.columns
        assert 'age_risk_interaction' in result.columns
        assert 'total_prior_utilization' in result.columns
        assert len(result.columns) > len(data.columns)
