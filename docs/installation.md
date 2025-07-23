# Installation Guide

## Prerequisites

- Python 3.8 or higher
- Git

## Installation Options

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

## Verification

Test your installation:

```bash
python -c "import src; print('Installation successful!')"
```

## Troubleshooting

Common issues and solutions:

1. **CUDA/GPU Issues**: If you encounter GPU-related errors, install CPU-only versions of PyTorch
2. **Memory Issues**: Reduce batch sizes in configuration files
3. **Missing Dependencies**: Ensure all requirements are installed with `pip install -r requirements.txt`
