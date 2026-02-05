# Time-and-Intervention Selection

**IMPORTANT NOTICE (February 2026)**: This repository previously contained implementation errors in the demonstration code that have now been corrected. Please use `acuityandintervention_CORRECTED.ipynb` instead of the original notebook. The published study results were not affected by these errors, as they were generated using separate production code with proper methodology.

---

The code here is intended to illustrate how to move from **static risk prediction** ("who is high-risk?") toward **operational decision support**:

- **When** to intervene (timing / acuity-aware triggers)
- **What** to do (intervention selection / next-best-action)

---

## What's in this repo

- **`acuityandintervention_CORRECTED.ipynb`** — CORRECTED end-to-end analysis notebook with proper temporal filtering (USE THIS)
- `acuityandintervention.ipynb` — Original notebook with known errors (archived for reference)
- `CORRECTED_CODE.py` — Standalone corrected implementation
- `CRITICAL_FIXES.md` — Detailed explanation of errors and corrections

**Important**: Use `acuityandintervention_CORRECTED.ipynb` for any new work. The original notebook contains temporal filtering errors that have been documented and corrected.

---

## Critical Corrections Applied

The corrected notebook (`acuityandintervention_CORRECTED.ipynb`) fixes two critical errors:

1. **Incorrect temporal cutoff date**: Changed from June 1, 2024 to December 31, 2024 (study training period end)
2. **Missing temporal filter in diagnosis features**: Added filtering to prevent data leakage by excluding future diagnoses

See `CRITICAL_FIXES.md` for detailed technical documentation.

---

## Conceptual overview (what the notebook is trying to do)

Traditional "risk scores" are often not actionable on their own. The notebook instead frames the workflow as a decision problem:

1. **Define acuity / urgency**  
   Construct an operational "acuity" signal (or multiple signals) that can be updated as new information arrives.

2. **Specify candidate interventions**  
   Enumerate a small menu of discrete actions (e.g., outreach modality, intensity, care pathway, referral).

3. **Estimate heterogeneity of response**  
   Learn which interventions work best for which patient strata (or contexts), rather than estimating only an average effect.

4. **Select time + intervention jointly**  
   Use the acuity signal (timing) plus estimated differential response (what) to produce a recommended next step.

---

## Important: Temporal Filtering for Prediction Models

This repository demonstrates time-to-event prediction modeling with proper temporal separation to prevent data leakage. Key principles:

1. **Training features must use only historical data** - no information from after the prediction timepoint
2. **Test set patients must be temporally separated** - enrolled after training data cutoff
3. **All data sources require temporal filtering** - diagnoses, events, interventions must be filtered by timestamp

The corrected notebook implements these principles correctly. The original notebook contained errors that violated these principles.

---

## Requirements

### Software
- Python 3.9+ (3.10/3.11 recommended)
- Jupyter (Notebook or JupyterLab)

### Common Python dependencies
The notebook requires a standard scientific Python stack. If you don't already have these installed:

```bash
pip install numpy pandas scikit-learn scipy statsmodels matplotlib jupyter
```

> If the notebook imports additional packages (e.g., causal inference libraries), install those as prompted by import errors.

---

## Quickstart

### 1) Create an environment (recommended)

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install jupyter numpy pandas scikit-learn scipy statsmodels matplotlib
```

### 2) Launch Jupyter and open the CORRECTED notebook

```bash
jupyter notebook
```

Then open `acuityandintervention_CORRECTED.ipynb` (not the original).

### 3) Adapt to your data

This code requires access to healthcare databases and is provided for illustration only. To adapt to your data:

1. Replace database connection strings with your own data sources
2. Ensure all temporal filtering is applied correctly (see corrected cells)
3. Verify timestamp fields exist for all data sources (diagnoses, events, interventions)
4. Run the analysis and verify no temporal leakage occurs

**Critical**: The corrected notebook shows the proper implementation of temporal filtering. Review the changes documented in `CRITICAL_FIXES.md` before adapting to your data.

---

## Data Requirements

This analysis requires:
- Patient demographics with enrollment/index dates
- Clinical diagnoses with timestamps
- Healthcare utilization events (ADT, encounters) with timestamps
- Intervention records with timestamps

**All data sources must include timestamp fields** to enable proper temporal filtering and prevent data leakage.

---

## File Guide

| File | Purpose | Status |
|------|---------|--------|
| `acuityandintervention_CORRECTED.ipynb` | Corrected analysis notebook | ✓ Use this |
| `acuityandintervention.ipynb` | Original notebook with errors | ⚠ Reference only |
| `CORRECTED_CODE.py` | Standalone corrected implementation | ✓ Use as reference |
| `CRITICAL_FIXES.md` | Technical documentation of corrections | ✓ Read for details |

---

## Citation

If you use methods from this repository, please cite:

Basu S, Patel SY, Sheth P, et al. Integrated acuity prediction and intervention selection for population health management. *Journal of the American Medical Informatics Association*. 2025.

---

## Acknowledgments

We thank the JAMIA editorial office for identifying implementation errors in the original demonstration code, enabling corrections before broader use.

---

## Contact

For questions about the methodology or implementation:
- Sanjay Basu, MD PhD: sanjay.basu@waymarkcare.com

For questions about the corrected code:
- See `CRITICAL_FIXES.md` for detailed documentation of all corrections
