#!/usr/bin/env python3
"""
Complete Population Health ML Analysis Pipeline

This script runs the complete analysis pipeline including data processing,
model training, evaluation, and result generation.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_engineering import FeatureEngineering
from src.data_processing.preprocessing import Preprocessing
from src.models.traditional_models import TraditionalModels
from src.models.ensemble_models import EnsembleModels
from src.models.deep_learning_models import DeepLearningModels
from src.models.causal_inference_models import CausalInferenceModels
from src.models.reinforcement_learning_models import ReinforcementLearningModels
from src.evaluation.metrics import Metrics
from src.evaluation.clinical_validation import ClinicalValidation
from src.visualization.figures import Figures
from src.visualization.tables import Tables
from src.utils.config import config
from src.utils.helpers import setup_logging, create_directories, save_results

# Set up logging
logger = logging.getLogger(__name__)


class PopulationHealthAnalysis:
    """Main analysis pipeline for population health ML study."""
    
    def __init__(self, data_path: str, output_dir: str, config_path: str = None):
        """Initialize the analysis pipeline.
        
        Args:
            data_path: Path to input data
            output_dir: Directory for output files
            config_path: Path to configuration file
        """
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Load configuration
        if config_path:
            config.load_config(config_path)
        
        # Set up directories
        create_directories(output_dir)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineering()
        self.preprocessor = Preprocessing()
        self.traditional_models = TraditionalModels()
        self.ensemble_models = EnsembleModels()
        self.deep_learning_models = DeepLearningModels()
        self.causal_models = CausalInferenceModels()
        self.rl_models = ReinforcementLearningModels()
        self.metrics = Metrics()
        self.clinical_validator = ClinicalValidation()
        self.figure_generator = Figures()
        self.table_generator = Tables()
        
        # Storage for results
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.models = {}
        self.results = {}
    
    def load_and_process_data(self):
        """Load and process the data."""
        logger.info("Loading and processing data...")
        
        # Load data
        self.data = self.data_loader.load_data(self.data_path)
        logger.info(f"Loaded data with {len(self.data)} patients")
        
        # Feature engineering
        self.data = self.feature_engineer.create_features(self.data)
        logger.info(f"Created features: {self.data.shape[1]} total features")
        
        # Preprocessing and splitting
        split_data = self.preprocessor.prepare_data(self.data)
        self.X_train, self.X_val, self.X_test = split_data['X_train'], split_data['X_val'], split_data['X_test']
        self.y_train, self.y_val, self.y_test = split_data['y_train'], split_data['y_val'], split_data['y_test']
        
        logger.info(f"Data split - Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")
    
    def train_all_models(self):
        """Train all machine learning models."""
        logger.info("Training all models...")
        
        # Traditional models
        logger.info("Training traditional models...")
        self.models.update(self.traditional_models.train_all_models(
            self.X_train, self.y_train, self.X_val, self.y_val
        ))
        
        # Ensemble models
        logger.info("Training ensemble models...")
        self.models.update(self.ensemble_models.train_all_models(
            self.X_train, self.y_train, self.X_val, self.y_val
        ))
        
        # Deep learning models
        logger.info("Training deep learning models...")
        self.models.update(self.deep_learning_models.train_all_models(
            self.X_train, self.y_train, self.X_val, self.y_val
        ))
        
        # Causal inference models
        logger.info("Training causal inference models...")
        self.models.update(self.causal_models.train_all_models(
            self.X_train, self.y_train, self.X_val, self.y_val
        ))
        
        # Reinforcement learning models
        logger.info("Training reinforcement learning models...")
        self.models.update(self.rl_models.train_all_models(
            self.X_train, self.y_train, self.X_val, self.y_val
        ))
        
        logger.info(f"Trained {len(self.models)} models total")
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        logger.info("Evaluating models...")
        
        # Evaluate on test set
        self.results = self.metrics.evaluate_all_models(
            self.models, self.X_test, self.y_test
        )
        
        # Calculate clinical utility metrics
        self.results = self.metrics.calculate_clinical_utility(
            self.models, self.X_test, self.y_test, self.results
        )
        
        # Perform clinical validation
        validation_results = self.clinical_validator.validate_models(
            self.models, self.X_test, self.y_test
        )
        self.results['clinical_validation'] = validation_results
        
        logger.info("Model evaluation complete")
    
    def generate_outputs(self):
        """Generate figures, tables, and reports."""
        logger.info("Generating outputs...")
        
        # Generate figures
        figures_dir = os.path.join(self.output_dir, 'figures')
        self.figure_generator.generate_all_figures(self.results, figures_dir)
        
        # Generate tables
        tables_dir = os.path.join(self.output_dir, 'tables')
        self.table_generator.generate_all_tables(self.results, tables_dir)
        
        # Save detailed results
        results_file = os.path.join(self.output_dir, 'complete_results.json')
        save_results(self.results, results_file)
        
        logger.info("Output generation complete")
    
    def save_models(self):
        """Save trained models."""
        logger.info("Saving models...")
        
        models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_file = os.path.join(models_dir, f'{model_name}.pkl')
            try:
                import joblib
                joblib.dump(model, model_file)
                logger.info(f"Saved {model_name}")
            except Exception as e:
                logger.warning(f"Could not save {model_name}: {e}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        start_time = time.time()
        logger.info("Starting complete population health ML analysis...")
        
        try:
            # Step 1: Load and process data
            self.load_and_process_data()
            
            # Step 2: Train all models
            self.train_all_models()
            
            # Step 3: Evaluate models
            self.evaluate_models()
            
            # Step 4: Generate outputs
            self.generate_outputs()
            
            # Step 5: Save models
            self.save_models()
            
            # Summary
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Analysis complete! Total time: {duration:.2f} seconds")
            
            # Print summary results
            self.print_summary()
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def print_summary(self):
        """Print summary of results."""
        logger.info("\n" + "="*50)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*50)
        
        # Data summary
        logger.info(f"Dataset: {len(self.data)} patients")
        logger.info(f"Features: {self.X_train.shape[1]}")
        logger.info(f"Models trained: {len(self.models)}")
        
        # Top performing models
        if 'performance_summary' in self.results:
            perf = self.results['performance_summary']
            logger.info("\nTop performing models (7-day AUC):")
            for i, (model, auc) in enumerate(perf['top_models_7d'][:5]):
                logger.info(f"{i+1}. {model}: {auc:.3f}")
        
        # Clinical utility
        if 'clinical_utility' in self.results:
            clinical = self.results['clinical_utility']
            logger.info(f"\nBest clinical utility (NNT): {clinical['best_nnt']:.2f}")
            logger.info(f"Clinical concordance: {clinical['concordance']:.3f}")
        
        logger.info(f"\nResults saved to: {self.output_dir}")
        logger.info("="*50)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run complete population health ML analysis"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=args.log_level)
    
    # Validate inputs
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Set number of jobs
    os.environ['N_JOBS'] = str(args.n_jobs)
    
    # Run analysis
    analysis = PopulationHealthAnalysis(
        data_path=args.data_path,
        output_dir=args.output_dir,
        config_path=args.config_path
    )
    
    analysis.run_complete_analysis()


if __name__ == "__main__":
    main()

