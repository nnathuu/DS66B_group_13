
# 30-Day Readmission Prediction for Diabetic Patients

> A group project for the **Statistics** course, applying machine learning techniques to a real-world healthcare problem: predicting whether a diabetic patient will be readmitted to the hospital within 30 days (`<30`) after discharge.

---

## Table of Contents

1. [Problem Introduction](#1-problem-introduction)
2. [Objectives](#2-objectives)
3. [Dataset](#3-dataset)
4. [Project Structure](#4-project-structure)
5. [Workflow](#5-workflow)
6. [Results](#6-results)
7. [Installation and Reproduction](#7-installation-and-reproduction)
8. [Team Contributions](#8-team-contributions)
9. [Team Git Workflow](#9-team-git-workflow)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [References](#11-references)

---

## 1. Problem Introduction

Early hospital readmission, defined as readmission within 30 days after discharge, is an important indicator of healthcare quality. For diabetic patients, readmission increases treatment costs and health risks. Many healthcare systems may also face additional costs or penalties when readmission rates are high.

If hospitals can identify high-risk patients early, they can intervene before discharge by providing additional counseling, scheduling follow-up appointments, and monitoring patients after discharge. These measures may help reduce readmissions and optimize healthcare resources.

**Problem statement:** Given information about a hospital encounter involving a diabetic patient, including demographics, diagnoses, laboratory tests, medications, and medical history, predict whether the patient will be readmitted within **30 days (`<30`)**.

- Problem type: **Binary classification**.
- Target variable: `readmitted`, converted into a binary variable: `<30` → **1**, while the remaining values (`>30`, `NO`) → **0**.
- Main challenges: **Class imbalance**, as readmissions within 30 days account for a relatively small proportion of cases; missing values; and categorical variables with a large number of distinct values, such as ICD-9 diagnosis codes.

## 2. Objectives

- Apply machine learning techniques learned in class to a real-world business or healthcare problem.
- Build a complete pipeline covering data understanding, cleaning, EDA, feature engineering, modeling, and evaluation.
- Comparison with existing approaches: Compare the proposed approach with existing implementations and published studies, including Emi-Johnson and Nkrumah (2025), to identify potential improvements in preprocessing, feature engineering, model selection, and evaluation.
- Complete a comprehensive report following the structure of a graduation project, including a literature review and a team contribution table, together with presentation slides. Charts and tables are prioritized for presenting results.

## 3. Dataset

**Source:** *Diabetes 130-US hospitals for years 1999–2008* (UCI Machine Learning Repository / Kaggle). The dataset contains clinical records from 130 hospitals and healthcare systems in the United States over a 10-year period.

| File | Description |
|---|---|
| `diabetic_data.csv` | Main dataset, where each row represents a hospital encounter; approximately 100,000 rows and 50 columns |
| `IDS_mapping.csv` | Mapping tables that translate codes into labels for `admission_type_id`, `discharge_disposition_id`, and `admission_source_id` |

The raw data is stored in `data/input/`. For detailed descriptions of individual columns, see [`data_dictionary.md`](data_dictionary.md).

**Main feature groups:**

- **Identifiers:** `encounter_id`, `patient_nbr`
- **Demographics:** `race`, `gender`, `age`, `weight`
- **Admission and discharge information:** `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `time_in_hospital`, `payer_code`, `medical_specialty`
- **Healthcare utilization:** `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`
- **Diagnoses:** `diag_1`, `diag_2`, `diag_3` (ICD-9 codes)
- **Laboratory tests:** `max_glu_serum`, `A1Cresult`
- **Diabetes medications:** medication-related columns (`metformin`, `insulin`, etc.), `change`, `diabetesMed`
- **Target variable:** `readmitted` (`<30`, `>30`, `NO`)

## 4. Project Structure

```text
diabetes-readmission-prediction/
├── data/                       # Stores train/test datasets and auxiliary csv files used throughout the project.
├── input/                      # Raw data: diabetic_data.csv, IDS_mapping.csv
├── notebooks/
│   ├── 01_data_eda.ipynb       # Data understanding, cleaning, EDA, basic feature engineering
│   └── 02_modeling.ipynb       # Advanced feature engineering & Model training
│   └── 03_evaluation.ipynb     # Evaluation
├── outputs/
|   ├── evaluation/             # Evaluation figures and tables
|   |   ├── figures/
|   |   └── tables/
│   └── figures/                # Figures from 01_data_eda
├── src/
│   ├── compare_models.py       # Compares model performance to support model selection
│   └── evaluation.py           # Evaluates the selected model using performance metrics and evaluation visualizations
├── data_dictionary.md          # Data dictionary
├── requirements.txt            # Required Python packages
├── .gitignore
└── README.md
```

> Note: The `data/` and `input/` directories are kept separate from the source code to prevent accidental modification of the raw data. All processing steps are implemented in `src/` and `notebooks/` to ensure reproducibility.

## 5. Workflow

### 5.1. Data Understanding

- Examine the dataset dimensions, data types, and the meaning of each column (see `data_dictionary.md`).
- Cross-reference ID codes with `IDS_mapping.csv` to understand the actual labels for admission types, discharge dispositions, and admission sources.
- Examine the target variable distribution and the degree of class imbalance.

### 5.2. Data Cleaning

- Replace the missing-value marker `?` with `NaN`.
- **Remove patients who died or were discharged to hospice care**, as they cannot be readmitted.
- Remove near-constant columns that contain almost exclusively one value and therefore provide little discriminative information.
- Handle missing values in `weight`, `payer_code`, `medical_specialty`, and `race` by assessing missingness rates and deciding whether to drop columns or assign labels such as `"Unknown"` or `"Missing"`, depending on the extent of missingness.
- Handle duplicate records at the patient level using `patient_nbr` to prevent information leakage between the training and test sets.

### 5.3. Preprocessing

- Convert the target variable into a binary label: `<30` → 1, all other values → 0.
- Group **ICD-9** diagnosis codes (`diag_1`, `diag_2`, `diag_3`) into broader disease categories, such as circulatory, respiratory, digestive, diabetes-related, injury, musculoskeletal and neurological, genitourinary, neoplasms, and other conditions.
- Group rare categories and convert ID codes into meaningful labels.

### 5.4. Exploratory Data Analysis (EDA)

- Examine the distributions of the target variable and numerical and categorical features.
- Investigate the relationships between features such as age, previous hospitalizations, length of stay, and medication counts and the 30-day readmission rate.
- Conduct basic statistical analyses, including correlation analysis and appropriate statistical tests, to support feature selection.
- Save visualizations in `outputs/figures/` for use in the report and presentation slides.

### 5.5. Feature Engineering

- Create aggregate features from healthcare utilization history, such as the total number of previous outpatient visits and hospitalizations.
- Engineer medication-related features, including the number of medications used and whether medication dosages have changed.
- Create features based on ICD-9 diagnosis groups.
- Encode categorical variables and scale numerical variables when necessary.

### 5.6. Modeling

- Split the data into training and test sets, preferably at the patient level while preserving class distributions through stratification.
- Establish a baseline model for comparison, followed by more advanced models using `scikit-learn` and `catboost`.
- Address class imbalance using techniques such as class weights and resampling, and tune hyperparameters.
- Implement the modeling workflow in `notebooks/02_modeling.ipynb`.

### 5.7. Evaluation

Because the dataset is imbalanced (about 11% positive), **accuracy is not used as the sole evaluation metric**. The following metrics are considered together:

- Average Precision (PR-AUC), the main metric for model selection, and ROC-AUC.
- Precision, Recall and F1-score for the positive <30 class, at the threshold chosen on the validation split.
- Confusion matrix at the final threshold.

Evaluation functions are implemented in `src/evaluation.py` and applied to `notebooks/03_evaluation.ipynb`.

## 6. Results

- **Model selection**: Eight candidates (Logistic Regression, Decision Tree, Random Forest and CatBoost, each with and without class weights) are compared with 3-fold patient-grouped cross-validation on the fit set and with a single patient-disjoint validation split. Average Precision (AP) is the main criterion, with ROC-AUC as a secondary one. CatBoost without class weights ranks first on both and is selected for tuning.
- **Final test performance**: The tuned CatBoost is evaluated once on the held-out test set at the threshold chosen on the validation split (0.1273). We report ROC-AUC, AP, Precision, Recall, Specificity and F1 with 95% bootstrap confidence intervals (ROC-AUC also with DeLong’s interval), together with the ROC and precision-recall curves and the confusion matrix.
- **Baseline comparison and statistical significance**: Each baseline is refitted with the same procedure as the final model and scored on the same test encounters, each at its own validation-selected threshold. Differences are tested with DeLong’s test (ROC-AUC), a paired bootstrap (AP and F1) and exact McNemar tests (recall and specificity), with Holm correction across the eight comparisons.
- **Comparison with published results**: The final model is compared with figures reported in published studies and public repositories on the same dataset. These figures are not re-run by us, and the splits, preprocessing and class balancing may differ (in particular, whether the split is grouped by patient), so the comparison is indicative and not a controlled one.

## 7. Installation and Reproduction

**Requirements:** Python 3.11 or later. Using a virtual environment is recommended.

```bash
# 1. Clone the repository
git clone https://github.com/nnathuu/diabetes-readmission-prediction.git
cd diabetes-readmission-prediction

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Main dependencies** (listed in `requirements.txt`): `pandas`, `numpy`, `scipy`, `scikit-learn`, `catboost`, `matplotlib`, `seaborn`, and `ipykernel`.

**How to run the project:**

1. Place `diabetic_data.csv` and `IDS_mapping.csv` in `input/` if they are not already available.
2. Open and run the notebooks in the following order:
   - `notebooks/01_data_eda.ipynb` — data cleaning, EDA, and feature engineering.
   - `notebooks/02_modeling.ipynb` — model training
   - `notebooks/03_evaluation.ipynb` — evaluation
3. The resulting figures are saved in `outputs/figures/`.

## 8. Team Contributions

> **TODO:** Complete the list of team members and their contribution percentages as required by the course.

| Team Member | Student ID | Responsibilities | Contribution |
|---|---|---|---|
| Bui Vi Anh (pipotht) | 11247122 | Literature review, evaluation metrics, statistical model comparison, error analysis, comparison, result visualization | 33.33% |
| Nguyen Ngo Anh Thu (nnathuu) | 11247229 | Data understanding, data cleaning and missing value handling, preprocessing, ICD-9 grouping, EDA, statistical analysis, and basic feature engineering | 33.33% |
| Nguyen Thai Anh Thu (Ngthaianhthu) | 11247230 | Advanced feature engineering, baseline models, advanced model, imbalance handling, hyperparameter tuning, model selection  | 33.33 |

## 9. Team Git Workflow

- Mixed Development Environments: The project began with a hybrid setup. Some team members developed locally and pushed directly to the repository, while others utilised Google Colab for data exploration and initial modelling.
- Iterative Synchronization: Because of these differing environments, early collaboration relied on direct commits and manual file updates—such as replacing older Colab notebooks—rather than a strict feature-branching or Pull Request review process.
- Consolidated Repository Management: As the machine learning pipeline and reporting stabilized, development consolidated around local Git workflows. Throughout all phases, large datasets, local virtual environments, and temporary files were strictly excluded using .gitignore to maintain a clean codebase.

## 10. Limitations and Future Work

### Limitations

- Weak associations: Highest correlation = 0.167; Cramér's V = 0.112.
- Limited EDA: Univariate analysis; plots show counts, not readmission rates.
- Information loss: Winsorization, broad ICD-9 grouping, and missing-value handling.
- Limited validation: One patient-level split; no external or temporal validation.
- Outdated data: 130 US hospitals (1999–2008); may not reflect current practices.
- Model limitations: One-hot encoding and limited tuning (20 configurations); no significant improvement over default CatBoost.
- Threshold and interpretation: F1-based threshold, assumed costs, and no SHAP or feature ablation.

### Future Work

- Robust evaluation: Repeated or nested patient-grouped cross-validation and external validation.
- Calibration and cost: Calibrate predictions and optimize thresholds using hospital-specific costs.
- Fairness: Decision curves, Recall@k, subgroup confidence intervals, and fairness analysis.
- Richer features: Longer utilization history, lab results, clinical notes, and medication changes.
- Model improvement: Native categorical CatBoost, alternative models, and SHAP analysis.

## 11. References

### References

- Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). [Diabetes 130-US Hospitals for Years 1999–2008](https://doi.org/10.24432/C5230J). UCI Machine Learning Repository.
- Strack et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates.*
- Centers for Medicare & Medicaid Services (CMS). *Hospital Readmissions Reduction Program (HRRP).*
- `scikit-learn` and `CatBoost` documentation.
For detailed methodology, analysis, and the complete references, please refer to the project report.
