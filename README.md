# Beyond Risk Scoring: Machine Learning Models for Personalized Intervention Selection in Population Health Management

This repository contains the complete code and analysis pipeline for the research paper "Beyond Risk Scoring: Machine Learning Models for Personalized Intervention Selection in Population Health Management" published in npj Digital Medicine.

## Overview

This study evaluates 17 state-of-the-art machine learning models for predicting healthcare utilization timing and optimizing personalized intervention selection in population health management. The analysis demonstrates that machine learning models can provide personalized intervention selection guidance that matches expert clinical judgment (κ = 0.82) while optimizing resource allocation based on individual patient characteristics.

## Key Findings

- **Personalized Intervention Selection**: Models achieved substantial clinical concordance (κ = 0.82) with expert clinician judgment
- **Precision Targeting**: 10-fold differences in intervention effectiveness based on patient characteristics (e.g., substance use support NNT 1.2 vs 12.3)
- **Risk-Stratified Allocation**: Appropriate resource concentration with 34% intervention rate in highest-risk quartile vs 12% in lowest-risk quartile
- **Time-to-Event Prediction**: DeepSurv achieved highest 7-day prediction performance (AUC 0.812, NNT 2.57)

## Repository Structure

```
population-health-ml-analysis/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
├── setup.py                          # Package installation
├── data/                             # Data directory
│   ├── README.md                     # Data documentation
│   └── sample_data_generator.py      # Generate synthetic data
├── src/                              # Source code
│   ├── data_processing/              # Data processing modules
│   ├── models/                       # Model implementations
│   ├── evaluation/                   # Evaluation metrics
│   ├── visualization/                # Plotting functions
│   └── utils/                        # Utility functions
├── scripts/                          # Analysis scripts
│   ├── run_full_analysis.py          # Complete analysis pipeline
│   ├── train_models.py               # Model training
│   ├── evaluate_models.py            # Model evaluation
│   ├── generate_figures.py           # Figure generation
│   └── clinical_validation.py        # Clinical validation
├── notebooks/                        # Jupyter notebooks
├── results/                          # Output directory
├── tests/                           # Unit tests
└── docs/                            # Documentation
```

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/population-health-ml-analysis.git
cd population-health-ml-analysis

# Create conda environment
conda env create -f environment.yml
conda activate pop-health-ml

# Install package in development mode
pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/population-health-ml-analysis.git
cd population-health-ml-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Generate Sample Data

Since the original data contains protected health information, we provide a synthetic data generator:

```bash
python data/sample_data_generator.py --n_patients 10000 --output_dir data/
```

### 2. Run Complete Analysis

```bash
python scripts/run_full_analysis.py --data_path data/synthetic_data.csv --output_dir results/
```

### 3. Generate Figures and Tables

```bash
python scripts/generate_figures.py --results_dir results/ --output_dir results/figures/
```

## Detailed Usage

### Data Processing

```python
from src.data_processing import DataLoader, FeatureEngineering, Preprocessing

# Load data
loader = DataLoader()
data = loader.load_data('data/synthetic_data.csv')

# Feature engineering
fe = FeatureEngineering()
features = fe.create_features(data)

# Preprocessing
preprocessor = Preprocessing()
X_train, X_test, y_train, y_test = preprocessor.prepare_data(features)
```

### Model Training

```python
from src.models import TraditionalModels, EnsembleModels, DeepLearningModels

# Train traditional models
traditional = TraditionalModels()
lr_model = traditional.train_logistic_regression(X_train, y_train)
cox_model = traditional.train_cox_model(X_train, y_train)

# Train ensemble models
ensemble = EnsembleModels()
rf_model = ensemble.train_random_forest(X_train, y_train)
xgb_model = ensemble.train_xgboost(X_train, y_train)

# Train deep learning models
deep = DeepLearningModels()
deepsurv_model = deep.train_deepsurv(X_train, y_train)
deephit_model = deep.train_deephit(X_train, y_train)
```

### Evaluation

```python
from src.evaluation import Metrics, ClinicalValidation

# Calculate performance metrics
metrics = Metrics()
results = metrics.evaluate_all_models(models, X_test, y_test)

# Clinical validation
validator = ClinicalValidation()
concordance = validator.validate_intervention_selection(models, validation_cases)
```

### Visualization

```python
from src.visualization import Figures, Tables

# Generate figures
fig_gen = Figures()
fig_gen.plot_roc_curves(results)
fig_gen.plot_intervention_effectiveness(results)

# Generate tables
table_gen = Tables()
table_gen.create_performance_table(results)
table_gen.create_clinical_utility_table(results)
```

## Models Implemented

### Traditional Methods (2 models)
- Logistic Regression with L2 regularization
- Cox Proportional Hazards with Elastic Net

### Ensemble Methods (3 models)
- Random Forest
- Random Survival Forest
- XGBoost

### Deep Learning (5 models)
- DeepSurv (Deep Cox Proportional Hazards)
- DeepHit (Competing Risks)
- SurvTRACE (Transformer-based Survival Analysis)
- Multi-Task Logistic Regression (MTLR)
- Deep Discrete-Time Survival Analysis (DDRSA)

### Causal Inference (5 models)
- T-learner
- X-learner
- Causal Forest
- Bayesian Causal Forest
- Generalized Random Forest

### Reinforcement Learning (2 models)
- Contextual Bandits (LinUCB)
- Deep Q-Networks

## Evaluation Metrics

- **Discrimination**: Area Under ROC Curve (AUC)
- **Clinical Utility**: Number Needed to Treat (NNT)
- **Performance**: Sensitivity, Specificity, Positive Predictive Value
- **Calibration**: Hosmer-Lemeshow test, Brier score
- **Clinical Concordance**: Cohen's kappa with expert clinicians

## Data Requirements

The analysis expects data with the following structure:

### Required Columns
- `patient_id`: Unique patient identifier
- `age`: Patient age in years
- `gender`: Patient gender (0/1 or M/F)
- `race_ethnicity`: Race/ethnicity categories
- `chronic_conditions`: Number of chronic conditions
- `prior_ed_visits`: Prior emergency department visits
- `prior_hospitalizations`: Prior hospitalizations
- `substance_use_disorder`: Binary indicator
- `mental_health_condition`: Binary indicator
- `housing_instability`: Binary indicator
- `outcome_7d`, `outcome_30d`, `outcome_90d`, `outcome_180d`: Binary outcomes
- `time_to_event`: Time to first event (days)
- `intervention_type`: Type of intervention received
- `intervention_completion`: Whether intervention was completed

### Optional Columns
- Additional clinical variables
- Social determinants of health
- Medication history
- Care coordinator assessments

## Reproducibility

All analyses include:
- Fixed random seeds for reproducibility
- Temporal data splitting to prevent data leakage
- Rigorous cross-validation procedures
- Bootstrap confidence intervals
- Comprehensive logging

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{basu2024beyond,
  title={Beyond Risk Scoring: Machine Learning Models for Personalized Intervention Selection in Population Health Management},
  author={Basu, Sanjay},
  journal={npj Digital Medicine},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Sanjay Basu, MD PhD
- **Email**: [your-email@domain.com]
- **Institution**: [Your Institution]

## Acknowledgments

- Waymark care coordination team
- Clinical reviewers for validation
- Population health management community

## Disclaimer

This code is provided for research purposes. The synthetic data generator creates realistic but artificial data for demonstration. Any clinical implementation should undergo appropriate validation and regulatory approval.

