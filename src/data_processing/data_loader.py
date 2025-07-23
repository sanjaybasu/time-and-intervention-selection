"""
Data loading and initial processing for population health analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and perform initial processing of population health data."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.required_columns = [
            'patient_id', 'age', 'gender', 'race_ethnicity',
            'chronic_conditions', 'prior_ed_visits', 'prior_hospitalizations',
            'composite_risk_score', 'outcome_7d', 'outcome_30d', 
            'outcome_90d', 'outcome_180d', 'time_to_event'
        ]
        
        self.optional_columns = [
            'substance_use_disorder', 'mental_health_condition',
            'housing_instability', 'intervention_type', 'intervention_completed',
            'dual_eligible', 'high_risk'
        ]
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        # Determine file type and load accordingly
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.parquet':
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded {len(data)} rows and {len(data.columns)} columns")
        
        # Validate data
        self._validate_data(data)
        
        # Initial cleaning
        data = self._initial_cleaning(data)
        
        return data
    
    def _validate_data(self, data: pd.DataFrame):
        """Validate that required columns are present.
        
        Args:
            data: Input DataFrame
        """
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info("Data validation passed")
    
    def _initial_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform initial data cleaning.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Performing initial data cleaning...")
        
        # Remove duplicates
        initial_rows = len(data)
        data = data.drop_duplicates(subset=['patient_id'])
        if len(data) < initial_rows:
            logger.info(f"Removed {initial_rows - len(data)} duplicate patients")
        
        # Handle missing values in key columns
        data = self._handle_missing_values(data)
        
        # Data type conversions
        data = self._convert_data_types(data)
        
        # Basic data validation
        data = self._validate_ranges(data)
        
        logger.info("Initial data cleaning complete")
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        # Log missing value patterns
        missing_summary = data.isnull().sum()
        missing_pct = (missing_summary / len(data)) * 100
        
        for col in missing_summary[missing_summary > 0].index:
            logger.info(f"Missing values in {col}: {missing_summary[col]} ({missing_pct[col]:.1f}%)")
        
        # Handle specific columns
        if 'gender' in data.columns:
            data['gender'] = data['gender'].fillna('Unknown')
        
        if 'race_ethnicity' in data.columns:
            data['race_ethnicity'] = data['race_ethnicity'].fillna('Unknown')
        
        # Fill numeric columns with median
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().sum() > 0:
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
                logger.info(f"Filled {col} missing values with median: {median_val}")
        
        # Fill binary columns with 0
        binary_columns = [col for col in data.columns if col.endswith('_disorder') or 
                         col.endswith('_condition') or col.endswith('_instability')]
        for col in binary_columns:
            if col in data.columns:
                data[col] = data[col].fillna(0)
        
        return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types appropriately.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with converted data types
        """
        # Convert patient_id to string
        data['patient_id'] = data['patient_id'].astype(str)
        
        # Convert binary outcomes to int
        outcome_columns = [col for col in data.columns if col.startswith('outcome_')]
        for col in outcome_columns:
            data[col] = data[col].astype(int)
        
        # Convert binary indicators to int
        binary_columns = [
            'substance_use_disorder', 'mental_health_condition', 
            'housing_instability', 'dual_eligible', 'high_risk',
            'intervention_completed'
        ]
        for col in binary_columns:
            if col in data.columns:
                data[col] = data[col].astype(int)
        
        # Ensure age is int
        data['age'] = data['age'].astype(int)
        
        # Ensure counts are int
        count_columns = ['chronic_conditions', 'prior_ed_visits', 'prior_hospitalizations']
        for col in count_columns:
            if col in data.columns:
                data[col] = data[col].astype(int)
        
        return data
    
    def _validate_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate that values are within expected ranges.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with validated ranges
        """
        # Age validation
        if 'age' in data.columns:
            invalid_age = (data['age'] < 0) | (data['age'] > 120)
            if invalid_age.sum() > 0:
                logger.warning(f"Found {invalid_age.sum()} patients with invalid age")
                data = data[~invalid_age]
        
        # Risk score validation
        if 'composite_risk_score' in data.columns:
            data['composite_risk_score'] = data['composite_risk_score'].clip(0, 1)
        
        # Time to event validation
        if 'time_to_event' in data.columns:
            data['time_to_event'] = data['time_to_event'].clip(0, 365)
        
        # Utilization validation
        for col in ['prior_ed_visits', 'prior_hospitalizations']:
            if col in data.columns:
                data[col] = data[col].clip(0, 50)  # Cap at reasonable maximum
        
        return data
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_patients': len(data),
            'n_features': len(data.columns),
            'date_range': {
                'start': data['data_generation_date'].min() if 'data_generation_date' in data.columns else None,
                'end': data['data_generation_date'].max() if 'data_generation_date' in data.columns else None
            },
            'demographics': {},
            'outcomes': {},
            'missing_data': {}
        }
        
        # Demographics
        if 'age' in data.columns:
            summary['demographics']['age'] = {
                'mean': data['age'].mean(),
                'std': data['age'].std(),
                'min': data['age'].min(),
                'max': data['age'].max()
            }
        
        if 'gender' in data.columns:
            summary['demographics']['gender'] = data['gender'].value_counts().to_dict()
        
        if 'race_ethnicity' in data.columns:
            summary['demographics']['race_ethnicity'] = data['race_ethnicity'].value_counts().to_dict()
        
        # Outcomes
        outcome_columns = [col for col in data.columns if col.startswith('outcome_')]
        for col in outcome_columns:
            summary['outcomes'][col] = {
                'event_rate': data[col].mean(),
                'n_events': data[col].sum()
            }
        
        # Missing data
        missing_summary = data.isnull().sum()
        summary['missing_data'] = {
            col: {'count': count, 'percentage': (count / len(data)) * 100}
            for col, count in missing_summary.items() if count > 0
        }
        
        return summary

