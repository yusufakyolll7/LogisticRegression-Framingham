Logistic Regression — CHD Risk Prediction 🫀
This notebook uses Logistic Regression to predict 10-year coronary heart disease (CHD) risk on a subset of the Framingham Heart Study dataset (binary target: TenYearCHD).

Goal: Build a compact ML workflow (EDA → preprocessing → modeling → evaluation) and assess performance on an imbalanced medical dataset.

🔹 Dataset & Target

File: framingham.csv

Typical features: age, sex, cigsPerDay, BPMeds, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose, ...

Target: TenYearCHD (0 = no event, 1 = event)

🔹 What We Did

EDA: basic structure check, missing values, distributions/correlations

Preprocessing: select features, handle missing values, encode if needed, optional scaling

Modeling: LogisticRegression (L2), train/test split (75/25), optional class_weight='balanced'

Evaluation: accuracy, precision, recall, f1, confusion matrix, ROC-AUC

Extras: decision threshold preview (trade-off between precision/recall)

🔹 Key Notes

Class imbalance is present; recall for the positive class can improve with class_weight or threshold tuning.

Coefficients provide feature impact direction (±) for interpretability.

ROC-AUC and confusion matrix are included for clear performance comparison.

🔹 Tech Stack

Python • pandas • numpy • scikit-learn • matplotlib • seaborn
