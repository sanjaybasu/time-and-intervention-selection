# Usage Guide

## Quick Start

### 1. Generate Sample Data

```bash
python data/sample_data_generator.py --n_patients 10000 --output_dir data/
```

### 2. Run Complete Analysis

```bash
python scripts/run_full_analysis.py --data_path data/synthetic_data_10000patients.csv --output_dir results/
```

### 3. Generate Figures

```bash
python scripts/generate_figures.py --results_dir results/ --output_dir results/figures/
```

## Advanced Usage

### Custom Configuration

Create a custom configuration file:

```yaml
# config.yml
models:
  time_horizons: [7, 30, 90]
  cv_folds: 3

traditional:
  logistic_regression:
    C: [0.1, 1.0, 10.0]
```

Run with custom config:

```bash
python scripts/run_full_analysis.py --data_path data.csv --config_path config.yml
```

### Individual Components

```python
from src.data_processing import DataLoader, FeatureEngineering
from src.models import TraditionalModels

# Load and process data
loader = DataLoader()
data = loader.load_data('data.csv')

fe = FeatureEngineering()
features = fe.create_features(data)

# Train models
models = TraditionalModels()
trained_models = models.train_all_models(X_train, y_train, X_val, y_val)
```

## Output Files

The analysis generates:

- `results/models/`: Trained model files
- `results/figures/`: Generated figures
- `results/tables/`: Performance tables
- `results/complete_results.json`: Detailed results
