#!/usr/bin/env python3
"""
Synthetic Data Generator for Population Health ML Analysis

This script generates synthetic data that mimics the structure and statistical
properties of the real Medicaid population health data used in the study,
while preserving patient privacy.
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic population health data for analysis."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the data generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_demographics(self, n_patients: int) -> pd.DataFrame:
        """Generate demographic data.
        
        Args:
            n_patients: Number of patients to generate
            
        Returns:
            DataFrame with demographic information
        """
        # Age distribution (mean 45.2, SD 16.8)
        ages = np.random.normal(45.2, 16.8, n_patients)
        ages = np.clip(ages, 18, 90).astype(int)
        
        # Gender (64.3% female)
        genders = np.random.choice(['F', 'M'], n_patients, p=[0.643, 0.357])
        
        # Race/ethnicity distribution
        race_ethnicity = np.random.choice(
            ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
            n_patients,
            p=[0.45, 0.25, 0.20, 0.06, 0.04]
        )
        
        # Dual eligibility (30%)
        dual_eligible = np.random.choice([0, 1], n_patients, p=[0.7, 0.3])
        
        return pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'age': ages,
            'gender': genders,
            'race_ethnicity': race_ethnicity,
            'dual_eligible': dual_eligible
        })
    
    def generate_clinical_data(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Generate clinical characteristics.
        
        Args:
            demographics: DataFrame with demographic data
            
        Returns:
            DataFrame with clinical information
        """
        n_patients = len(demographics)
        
        # Chronic conditions (mean 3.7, SD 2.1)
        chronic_conditions = np.random.poisson(3.7, n_patients)
        chronic_conditions = np.clip(chronic_conditions, 0, 15)
        
        # Prior utilization (correlated with age and chronic conditions)
        age_factor = (demographics['age'] - 18) / 72  # Normalize age
        chronic_factor = chronic_conditions / 15  # Normalize chronic conditions
        
        # Prior ED visits (mean 1.8, SD 2.4)
        ed_base_rate = 1.8 + 2.0 * age_factor + 3.0 * chronic_factor
        prior_ed_visits = np.random.poisson(ed_base_rate)
        prior_ed_visits = np.clip(prior_ed_visits, 0, 20)
        
        # Prior hospitalizations (mean 0.6, SD 1.2)
        hosp_base_rate = 0.6 + 1.0 * age_factor + 1.5 * chronic_factor
        prior_hospitalizations = np.random.poisson(hosp_base_rate)
        prior_hospitalizations = np.clip(prior_hospitalizations, 0, 10)
        
        # Composite risk score (0-1 scale, mean 0.23, SD 0.18)
        risk_scores = np.random.beta(2, 6, n_patients)  # Beta distribution
        risk_scores = np.clip(risk_scores, 0, 1)
        
        # High-risk designation (top 20%)
        high_risk_threshold = np.percentile(risk_scores, 80)
        high_risk = (risk_scores >= high_risk_threshold).astype(int)
        
        # Specific conditions
        substance_use_disorder = np.random.choice([0, 1], n_patients, p=[0.85, 0.15])
        mental_health_condition = np.random.choice([0, 1], n_patients, p=[0.70, 0.30])
        housing_instability = np.random.choice([0, 1], n_patients, p=[0.88, 0.12])
        
        return pd.DataFrame({
            'chronic_conditions': chronic_conditions,
            'prior_ed_visits': prior_ed_visits,
            'prior_hospitalizations': prior_hospitalizations,
            'composite_risk_score': risk_scores,
            'high_risk': high_risk,
            'substance_use_disorder': substance_use_disorder,
            'mental_health_condition': mental_health_condition,
            'housing_instability': housing_instability
        })
    
    def generate_outcomes(self, demographics: pd.DataFrame, 
                         clinical: pd.DataFrame) -> pd.DataFrame:
        """Generate outcome data for different time horizons.
        
        Args:
            demographics: DataFrame with demographic data
            clinical: DataFrame with clinical data
            
        Returns:
            DataFrame with outcome information
        """
        n_patients = len(demographics)
        
        # Base event rates by time horizon
        base_rates = {
            7: 0.08,    # 8% at 7 days
            30: 0.15,   # 15% at 30 days
            90: 0.25,   # 25% at 90 days
            180: 0.35   # 35% at 180 days
        }
        
        # Risk factors
        age_factor = (demographics['age'] - 18) / 72
        risk_factor = clinical['composite_risk_score']
        chronic_factor = clinical['chronic_conditions'] / 15
        
        outcomes = {}
        times_to_event = []
        
        for horizon in [7, 30, 90, 180]:
            # Adjust event probability based on risk factors
            event_prob = base_rates[horizon] * (
                1 + 0.5 * age_factor + 
                2.0 * risk_factor + 
                0.3 * chronic_factor
            )
            event_prob = np.clip(event_prob, 0, 0.8)  # Cap at 80%
            
            # Generate binary outcomes
            outcomes[f'outcome_{horizon}d'] = np.random.binomial(1, event_prob)
        
        # Generate time to first event
        for i in range(n_patients):
            if outcomes['outcome_7d'][i]:
                time_to_event = np.random.uniform(1, 7)
            elif outcomes['outcome_30d'][i]:
                time_to_event = np.random.uniform(8, 30)
            elif outcomes['outcome_90d'][i]:
                time_to_event = np.random.uniform(31, 90)
            elif outcomes['outcome_180d'][i]:
                time_to_event = np.random.uniform(91, 180)
            else:
                time_to_event = 180  # Censored at 180 days
            
            times_to_event.append(time_to_event)
        
        outcomes['time_to_event'] = times_to_event
        
        return pd.DataFrame(outcomes)
    
    def generate_interventions(self, demographics: pd.DataFrame,
                              clinical: pd.DataFrame) -> pd.DataFrame:
        """Generate intervention data.
        
        Args:
            demographics: DataFrame with demographic data
            clinical: DataFrame with clinical data
            
        Returns:
            DataFrame with intervention information
        """
        n_patients = len(demographics)
        
        # Intervention types
        intervention_types = [
            'substance_use_support',
            'mental_health_support', 
            'housing_assistance',
            'medication_adherence',
            'care_coordination'
        ]
        
        # Intervention assignment based on patient characteristics
        interventions = []
        completion_rates = []
        
        for i in range(n_patients):
            # Determine intervention type based on patient needs
            if clinical['substance_use_disorder'][i] and np.random.random() < 0.7:
                intervention = 'substance_use_support'
                completion_rate = 0.78
            elif clinical['mental_health_condition'][i] and np.random.random() < 0.6:
                intervention = 'mental_health_support'
                completion_rate = 0.73
            elif clinical['housing_instability'][i] and np.random.random() < 0.5:
                intervention = 'housing_assistance'
                completion_rate = 0.69
            elif clinical['chronic_conditions'][i] >= 3 and np.random.random() < 0.4:
                intervention = 'medication_adherence'
                completion_rate = 0.71
            elif clinical['high_risk'][i] and np.random.random() < 0.3:
                intervention = 'care_coordination'
                completion_rate = 0.68
            else:
                intervention = 'none'
                completion_rate = 0.0
            
            interventions.append(intervention)
            
            # Generate completion status
            if intervention != 'none':
                completed = np.random.binomial(1, completion_rate)
            else:
                completed = 0
            
            completion_rates.append(completed)
        
        return pd.DataFrame({
            'intervention_type': interventions,
            'intervention_completed': completion_rates
        })
    
    def generate_additional_features(self, n_patients: int) -> pd.DataFrame:
        """Generate additional features for model training.
        
        Args:
            n_patients: Number of patients
            
        Returns:
            DataFrame with additional features
        """
        # Generate correlated features that might be useful for prediction
        features = {}
        
        # Medication-related features
        features['total_medications'] = np.random.poisson(5, n_patients)
        features['high_risk_medications'] = np.random.poisson(1.2, n_patients)
        
        # Utilization patterns
        features['ed_visit_frequency'] = np.random.exponential(0.3, n_patients)
        features['specialist_visits'] = np.random.poisson(2.1, n_patients)
        
        # Social determinants
        features['transportation_barriers'] = np.random.choice([0, 1], n_patients, p=[0.75, 0.25])
        features['social_support_score'] = np.random.uniform(0, 10, n_patients)
        
        # Care coordination metrics
        features['care_gaps'] = np.random.poisson(1.5, n_patients)
        features['provider_communication_score'] = np.random.uniform(1, 5, n_patients)
        
        return pd.DataFrame(features)
    
    def generate_dataset(self, n_patients: int = 10000) -> pd.DataFrame:
        """Generate complete synthetic dataset.
        
        Args:
            n_patients: Number of patients to generate
            
        Returns:
            Complete synthetic dataset
        """
        logger.info(f"Generating synthetic dataset with {n_patients} patients...")
        
        # Generate each component
        demographics = self.generate_demographics(n_patients)
        clinical = self.generate_clinical_data(demographics)
        outcomes = self.generate_outcomes(demographics, clinical)
        interventions = self.generate_interventions(demographics, clinical)
        additional = self.generate_additional_features(n_patients)
        
        # Combine all data
        dataset = pd.concat([
            demographics,
            clinical,
            outcomes,
            interventions,
            additional
        ], axis=1)
        
        # Add metadata
        dataset['data_generation_date'] = datetime.now().strftime('%Y-%m-%d')
        dataset['synthetic_data'] = True
        
        logger.info("Synthetic dataset generation complete!")
        
        return dataset
    
    def save_dataset(self, dataset: pd.DataFrame, output_path: str):
        """Save dataset to file.
        
        Args:
            dataset: Generated dataset
            output_path: Path to save the dataset
        """
        dataset.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
        
        # Print summary statistics
        logger.info("\nDataset Summary:")
        logger.info(f"Number of patients: {len(dataset)}")
        logger.info(f"Number of features: {len(dataset.columns)}")
        logger.info(f"7-day event rate: {dataset['outcome_7d'].mean():.3f}")
        logger.info(f"30-day event rate: {dataset['outcome_30d'].mean():.3f}")
        logger.info(f"90-day event rate: {dataset['outcome_90d'].mean():.3f}")
        logger.info(f"180-day event rate: {dataset['outcome_180d'].mean():.3f}")
        logger.info(f"High-risk patients: {dataset['high_risk'].mean():.3f}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic population health data"
    )
    parser.add_argument(
        "--n_patients",
        type=int,
        default=10000,
        help="Number of patients to generate (default: 10000)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
        help="Output directory (default: data/)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticDataGenerator(random_state=args.random_state)
    
    # Generate dataset
    dataset = generator.generate_dataset(n_patients=args.n_patients)
    
    # Save dataset
    output_path = f"{args.output_dir}/synthetic_data_{args.n_patients}patients.csv"
    generator.save_dataset(dataset, output_path)


if __name__ == "__main__":
    main()

