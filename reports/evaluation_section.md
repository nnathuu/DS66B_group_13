# Evaluation (draft for the report - all numbers come from `notebooks/03_evaluation.ipynb`)

> Figures live in `outputs/evaluation/figures/`, tables in `outputs/evaluation/tables/`.
> Items marked **[CHECK]** must be verified by the team before submission.

## 1. Evaluation protocol

The data were split once, by `patient_nbr` (80/20, `GroupShuffleSplit`, seed 42), so that no patient appears in both train and test (19,802 test encounters, 2,216 positives, prevalence 11.2%). Model selection used patient-grouped 3-fold cross-validation on the training set; the decision threshold was chosen on a patient-grouped validation split (maximum F1) and the test set was used exactly once. Because only 11% of encounters are positive, accuracy is uninformative (a classifier that flags nobody is 88.8% accurate). We therefore select models by **Average Precision (AP)**, a threshold-free metric whose random baseline equals the prevalence, and report **ROC-AUC, Recall, Precision, F1, MCC** at the operating threshold. Recall is emphasised because a missed readmission is costlier than a false alarm. All intervals are 95% bootstrap confidence intervals (1,000 resamples; DeLong for AUC).

## 2. Main result

| Metric | Tuned CatBoost (threshold 0.127) |
|---|---|
| ROC-AUC | 0.676 [0.664, 0.688] |
| Average Precision | 0.243 [0.229, 0.261] (2.17x random) |
| Recall | 0.508 [0.488, 0.528] |
| Precision | 0.197 [0.187, 0.208] |
| Specificity | 0.739 [0.733, 0.745] |
| F1 | 0.284 [0.272, 0.297] |

At this threshold the model flags 28.8% of patients (1,125 true positives, 4,586 false positives, 1,091 false negatives, 13,000 true negatives). Performance is clearly better than chance but modest in absolute terms, consistent with the literature on this dataset, where the available administrative variables carry limited information about post-discharge factors (social support, adherence, follow-up care).

*Figures: `01_roc_pr.png`, `01_confusion_matrix.png`, `01_score_distribution.png`.*

## 3. Model comparison and statistical testing

All nine models were re-fit with one common protocol and scored on the same test rows (`03_model_comparison_test.csv`, `03_model_comparison.png`). Differences were tested against the tuned CatBoost with DeLong's test (ROC-AUC), paired bootstrap (AP, F1; 2,000 resamples) and exact McNemar tests on recall and specificity, all Holm-corrected (`03_pairwise_tests_vs_tuned_catboost.csv`).

* The tuned CatBoost has a significantly higher AUC and AP than logistic regression (ΔAUC = +0.010, 95% CI [+0.004, +0.015]; ΔAP = +0.023), random forest (ΔAUC = +0.009; ΔAP = +0.020) and decision trees (ΔAUC ≈ +0.03; ΔAP ≈ +0.04). The margins over linear and bagged baselines are, however, small in absolute terms.
* **Tuning did not produce a statistically detectable gain** over the untuned CatBoost (ΔAUC = +0.001, p = 0.46; ΔAP = +0.001, p = 0.68; ΔF1 = +0.002, p = 0.91). Its only visible difference is a higher recall at the chosen threshold (0.508 vs 0.453), i.e. a different operating point rather than a better ranking.
* Class weighting did not improve ranking quality (tuned CatBoost vs. CatBoost + class weight: ΔAUC = +0.003, n.s.; ΔAP = +0.011, in favour of the tuned model), and it moves the F1-optimal threshold from about 0.13 to about 0.53, so its predicted probabilities are no longer calibrated.
* We did not use a t-test on the 3 CV folds: with 2 degrees of freedom it has almost no power. Paired tests on the shared test set are used instead.

## 4. Threshold selection and cost-sensitive analysis

The default threshold of 0.5 flags only 0.4% of patients (recall 2.4%) and is unusable. The validation-selected threshold (0.127) gives recall 0.51 / precision 0.20; lowering it to reach 70% recall requires a threshold of 0.096, flagging 48.7% of patients at 16.1% precision (`04_operating_points.csv`, `04_threshold_sweep.png`). The F1-optimal threshold on the test set itself (0.155) is slightly higher than the validation choice, which shows that the F1 optimum is flat and the choice of threshold is a policy decision more than a statistical one.

Assuming a missed readmission costs *k* times an unnecessary follow-up (`04_cost_sensitive_optimum.csv`, `04_cost_curves.png`):

| FN : FP cost | Optimal threshold | Recall | Flagged | Saving vs. best trivial policy |
|---|---|---|---|---|
| 2 | 0.326 | 9.0% | 2.2% | 3.8% |
| 5 | 0.189 | 26.4% | 10.4% | 13.0% |
| 10 | 0.087 | 76.3% | 56.0% | 16.7% |
| 20 | 0.051 | 98.1% | 93.2% | 2.5% |

The model adds value mainly when a missed readmission is roughly 5–10 times as costly as a wasted intervention. When the cost ratio is close to 1 or extremely large, the preferred policy may be to flag nobody or everybody, respectively. The empirical optimal thresholds lie close to the theoretical threshold \(1/(1+k)\) (e.g., 0.087 versus 0.091 for \(k=10\)). However, this agreement alone does not establish good probability calibration, which should be assessed separately using calibration curves and appropriate metrics. Decision-curve analysis (`04_decision_curve.png`) shows a positive net benefit over both trivial policies for threshold probabilities from approximately 4.5% to 40%, the upper end of the evaluated grid.

The cost ratios used in this analysis are illustrative assumptions rather than empirically established healthcare costs. They demonstrate how different trade-offs between missed readmissions and unnecessary interventions affect the optimal decision threshold. Future work should replace these assumptions with estimates from published health-economic studies or local hospital data. Therefore, the results should be interpreted as scenario-based decision analysis rather than a validated clinical policy.

## 5. Operational view

Ranking patients by predicted risk (`05_recall_at_topk.csv`, `05_gain_lift.png`): the top 10% of patients have a 28.7% readmission rate (lift 2.56) and contain 25.6% of all readmissions; the top 20% contain 39.8% and the top 30% contain 51.9%. The highest-risk 1% has 56% precision (lift 5.0).

## 6. Calibration

The model is well calibrated (`07_calibration.png`): ECE = 0.005, mean predicted risk 0.114 vs observed 0.112, calibration slope 1.08. The decile-wise observed rates (3.8% to 28.7%) track the predicted rates closely, so predicted probabilities can be read as risks, not only as scores.

## 7. Error analysis

(`06_*` tables and figures.) Prior inpatient stays dominate both error types.

* **Missed readmissions (FN) vs. caught ones (TP):** FN patients have had far fewer prior inpatient visits (mean 0.17 vs. 2.28; Cohen's d = -1.17) and total prior visits (0.56 vs. 3.47). 64% of missed cases were discharged home (disposition 1), versus 34% of caught cases. The model therefore struggles on "first-time-looking" patients whose readmission is not foreshadowed by utilisation history.
* **False alarms (FP) vs. true negatives (TN):** FP patients also have many prior inpatient stays (1.72 vs. 0.14; d = 1.72), longer stays and more diagnoses.
* Because one global threshold (0.127) is below the base risk of high-utilisers, the model behaves almost like a switch on prior inpatient count: patients with no prior inpatient stay are flagged 8.4% of the time (recall 20.1%), whereas patients with two or more prior stays are flagged 96-99% of the time (recall 99-100%, precision only 17-27%). The alert is therefore not selective inside the high-utilisation group, where within-group AUC is only 0.59-0.64. A tiered strategy (a higher threshold for high-utilisers) would be an obvious improvement.

## 8. Subgroup performance

(`07_subgroup_*.png`, `07_subgroup_performance.csv`; groups with fewer than 500 rows or 50 positives excluded.) Discrimination is similar for the two largest racial groups (Caucasian AUC 0.676, African-American 0.664; recall 0.512 vs 0.508) and for gender (AUC 0.681 female vs 0.671 male), although recall is higher for women (0.540 vs 0.471). Performance is weakest for the oldest patients (age 90-100: AUC 0.527, 95% CI [0.447, 0.607], essentially chance) and for neoplasm (AUC 0.61, recall 0.25) and genitourinary (AUC 0.61) primary diagnoses; it is strongest for patients aged 30-40 and 50-60 and for musculoskeletal and diabetes diagnoses. Differences in recall and false-positive rates between groups partly reflect different base rates under one global threshold and should be read as areas to monitor, not as proof of unfair treatment.

## 9. Comparison with public baselines

| Source | Model | ROC-AUC | AP | Same patient-grouped split? |
|---|---|---|---|---|
| Emi-Johnson & Nkrumah (2025, Cureus) | XGBoost / LogReg / RF | 0.667 / 0.642 / 0.630 | - | No patient grouping reported |
| GitHub chaitra-satheesh | LogReg + RandomOverSampler | 0.649 | 0.209 | Not verified |
| GitHub brunoarine/diabetes | RF / LogReg | 0.66 / 0.64 | - | Not verified |
| **This work** | **Tuned CatBoost** | **0.676** | **0.243** | **Yes (grouped by patient)** |

These figures were taken from the cited sources and were not independently reproduced. Differences in data splitting, preprocessing, and positive-class definitions may affect comparability. The controlled comparison within this study is presented in Section 3.

Our contribution relative to typical public notebooks is (i) a patient-grouped split and grouped cross-validation (CV), which reduce the risk of optimistic evaluation caused by repeated encounters from the same patient appearing across different data splits; (ii) average precision (AP)-based model selection and a validation-selected decision threshold; (iii) confidence intervals and statistical significance tests; and (iv) calibration, cost-sensitive, error, and subgroup analyses. Because the evaluation protocols of the external baselines have not all been verified, direct methodological comparisons should be interpreted cautiously.

No specific Kaggle notebooks were prescribed in the assignment. Therefore, the comparison uses the public baselines currently listed in Section 8 of the notebook. These results are descriptive rather than a controlled benchmark because datasets, split strategies, preprocessing and positive-class definitions may differ.

## 10. Limitations

Absolute discrimination is modest (AUC 0.68); the data (1999-2008, 130 US hospitals) lack social and post-discharge variables and may not reflect current practice. Test encounters from the same patient are correlated, so encounter-level bootstrap intervals are slightly too narrow (a patient-level cluster bootstrap is supported once test patient ids are saved). The F1-based threshold is a single operating point; real deployment needs a hospital-specific cost ratio and a capacity limit. Subgroup results on small groups are noisy. No external validation set was available.
