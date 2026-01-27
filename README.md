# Time-and-Intervention Selection 

The code here is intended to illustrate how to move from **static risk prediction** (“who is high-risk?”) toward **operational decision support**:

- **When** to intervene (timing / acuity-aware triggers)
- **What** to do (intervention selection / next-best-action)

---

## What’s in this repo

- `acuityandintervention.ipynb` — end-to-end analysis notebook for acuity modeling and intervention selection. :contentReference[oaicite:1]{index=1}

---

## Conceptual overview (what the notebook is trying to do)

Traditional “risk scores” are often not actionable on their own. The notebook instead frames the workflow as a decision problem:

1. **Define acuity / urgency**  
   Construct an operational “acuity” signal (or multiple signals) that can be updated as new information arrives.

2. **Specify candidate interventions**  
   Enumerate a small menu of discrete actions (e.g., outreach modality, intensity, care pathway, referral).

3. **Estimate heterogeneity of response**  
   Learn which interventions work best for which patient strata (or contexts), rather than estimating only an average effect.

4. **Select time + intervention jointly**  
   Use the acuity signal (timing) plus estimated differential response (what) to produce a recommended next step.

---

## Requirements

### Software
- Python 3.9+ (3.10/3.11 recommended)
- Jupyter (Notebook or JupyterLab)

### Common Python dependencies
The notebook typically requires a standard scientific Python stack. If you don’t already have these installed, start with:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `statsmodels`
- `matplotlib`
- `jupyter` / `jupyterlab`

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
