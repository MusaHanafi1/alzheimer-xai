# Explainable Dementia Classification
### A reproducible machine learning evaluation by Musa Hanafi

**Python · scikit-learn · XGBoost · Streamlit · SHAP**

A portfolio project demonstrating how to audit an existing ML prototype, correct evaluation leakage, compare models, and produce results that can be inspected and reproduced.

> **Synthetic-data demonstration.** The included run uses synthetic data. It classifies whether Clinical Dementia Rating (CDR) is greater than zero at the same visit. These results do not establish clinical effectiveness, Alzheimer's diagnosis, or future dementia risk.

## Download and explore

**[Download the complete revised project ZIP](https://raw.githubusercontent.com/MusaHanafi1/alzheimer-xai/main/alzheimer-xai-reproducible.zip)**

The repository currently distributes the project as a ZIP. Extract it to access the code, notebooks, Streamlit app, generated report, and supporting documentation.

**Start without installing Python:** after extracting, open `alzheimer-xai-reproducible/reports/demo/report.html` in your browser. The report includes model comparisons, calibration, error analysis, and an interactive decision-threshold explorer.

## What this project demonstrates

- **Data integrity:** explicit synthetic provenance, input SHA-256, schema checks, and a defined cohort.
- **Participant separation:** one latest observed visit per subject and disjoint development/test subjects.
- **Leakage control:** median imputation and scaling fitted inside training folds.
- **Model selection:** prevalence baseline, logistic regression, random forest, and XGBoost compared using development cross-validation.
- **Probability evaluation:** training-only sigmoid calibration, Brier score, and calibration plots.
- **Decision tradeoffs:** threshold chosen from development out-of-fold predictions, with held-out precision and recall reported separately.
- **Interpretability:** development-data permutation importance and a scoped SHAP example.
- **Reproducibility:** pinned dependencies, saved metrics, split hashes, an executable training entry point, and automated tests.

## Verified synthetic benchmark

The included CSV contains **317 synthetic visit records for 150 subjects**. Keeping the latest visit per subject yields **120 development subjects and 30 held-out subjects**, with seed 42. The holdout contains 12 positive and 18 negative examples.

Development CV selected **XGBoost**, followed by prespecified sigmoid calibration. The decision threshold was **0.42**, selected using development data.

| Metric | Final calibrated holdout result | Conditional 95% bootstrap interval |
|---|---:|---:|
| ROC-AUC | 0.912 | 0.768–1.000 |
| Average precision | 0.925 | 0.813–1.000 |
| Accuracy | 0.900 | 0.800–1.000 |
| Precision | 0.909 | 0.750–1.000 |
| Recall | 0.833 | 0.583–1.000 |
| F1 | 0.870 | 0.700–1.000 |
| Brier score | 0.081 | 0.020–0.152 |

**Confusion counts:** 17 true negatives, 1 false positive, 2 false negatives, and 10 true positives.

Intervals use 1,000 stratified test-subject bootstrap resamples with a fixed trained model and fixed class counts. They do not include retraining uncertainty. The small test set produces wide intervals.

Exact values and evaluation settings are stored in `reports/demo/metrics.json` inside the ZIP. Earlier score claims in the original archive are not treated as reproduced or used as a comparable baseline.

## Verification completed

- **13 automated tests passed**, including subject separation, predictor allowlisting, training-only imputation, saved-model consistency, SHAP local additivity, and Streamlit page/prediction interactions.
- A second seeded training run produced **identical metrics and held-out predictions**.
- The browser report's threshold explorer was checked interactively.

The package contains a GitHub Actions workflow ready for use after the source is extracted into a repository. A ZIP-only upload does not activate that workflow.

## Run locally

Extract the ZIP and open a terminal in the folder containing `train.py`. Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Then install, verify, train, and launch:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --shap
python -m streamlit run streamlit_app/app.py
```

The Streamlit app supports run overview, evaluation, a synthetic example prediction, interpretability, and methods/limitations. It is supplied for local use; no public Streamlit deployment is claimed.

## Inside the revised package

| File or folder | Purpose |
|---|---|
| `train.py` | Reproducible benchmark entry point |
| `dementia_ml/` | Data validation, modeling, evaluation, and reporting |
| `streamlit_app/app.py` | Interactive synthetic-data demonstration |
| `notebooks/01_benchmark.ipynb` | Walkthrough of the generated run |
| `tests/` | Integrity, saved-artifact, and app interaction checks |
| `data/demo/` | Explicitly synthetic input dataset |
| `reports/demo/` | Report, charts, metrics, model, and split manifest |
| `docs/MODEL_CARD.md` | Intended use, evaluation protocol, and limitations |
| `docs/AUDIT.md` | Findings from the original prototype and corrections |
| `docs/INTERVIEW_GUIDE.md` | Explanation of methodological choices |
| `QUICKSTART.md` | Setup and reproduction instructions |

## Important limitations

- The synthetic data generator was not present in the original archive; fidelity to real OASIS data is not independently verified.
- This is **same-visit CDR classification**, not future progression or confirmed Alzheimer's pathology.
- MMSE and CDR measure related cognitive constructs, which limits interpretation of the task.
- Development model-selection scores and threshold diagnostics are not fully nested unbiased performance estimates. The final holdout was excluded from model, calibration, and threshold choices.
- Small subgroup summaries do not establish fairness.
- Clinical usefulness would require a separately declared study on appropriately authorized real data and independent validation.

## Author and project history

**Musa Hanafi**  
MS, Data Analytics — University of Houston–Downtown  
MS, Artificial Intelligence — Indiana Wesleyan University, in progress

[LinkedIn](https://www.linkedin.com/in/musa-hanafi) · [GitHub](https://github.com/MusaHanafi1) · [ORCID](https://orcid.org/0009-0000-9155-2414)

The original project is retained as `alzheimer-xai_5.zip`. The revised package is **`alzheimer-xai-reproducible.zip`**.

This revision used AI assistance for implementation, auditing, documentation, and verification. It is a portfolio engineering artifact, not a new clinical research publication.

The original MIT code license is retained inside the package. Dataset provenance and rights are separate from the code license.
