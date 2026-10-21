# Evaluation module - how to run

```bash
pip install catboost          # missing from requirements.txt (only needed to re-fit models)
jupyter lab                   # or open notebooks/03_evaluation.ipynb in VS Code, kernel = your venv
```
1. Open `notebooks/03_evaluation.ipynb` and **Run All** (about 6-10 min on a laptop; the slow step is the 2,000-resample paired bootstrap).
2. `outputs/evaluation/model_predictions.csv` is cached. To re-fit all models: `python src/compare_models.py --force`.
3. Optional (more rigorous CIs): in `01_data_eda.ipynb`, before `patient_nbr` is dropped from `test_df`, add
   `test_df[["patient_nbr"]].to_csv("../data/test_patient_groups.csv", index=False)`.
   The notebook then automatically uses a patient-level cluster bootstrap.

Files: `src/evaluation.py` (all metric / test / plot functions), `src/compare_models.py` (common refit protocol),
`notebooks/03_evaluation.ipynb`, `reports/evaluation_section.md` (English draft text).
