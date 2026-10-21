# Customer Churn Prediction and Retention Intelligence

## 1. Project Overview
This project predicts customer churn using machine-learning classification models and provides customer risk categories with suggested retention actions.

## 2. Dataset
- **Dataset:** Telco Customer Churn
- **Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Target column:** `Churn` (`Yes` = churn, `No` = did not churn)
- **Dataset size used:** 7,043 rows and 21 columns before preprocessing.

The dataset includes customer tenure, contract type, services, billing information, and churn status. Check the dataset page for its current usage terms and attribution requirements.

## 3. Project Objectives
- Explore customer data and identify churn patterns.
- Clean and preprocess numeric and categorical features.
- Train and compare three classification models.
- Evaluate predictions with multiple metrics and confusion matrices.
- Analyze prediction errors.
- Estimate churn risk and suggest retention actions.

## 4. Machine-Learning Models
The notebook trains:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The current model-selection code selects the model with the highest macro-F1 score on the held-out test set. The selected model in the recorded run was **Gradient Boosting**.

## 5. Methodology
1. Load and inspect the dataset.
2. Check missing values and duplicate rows.
3. Clean the data and examine churn distribution.
4. Explore churn patterns using charts.
5. Separate input features (`X`) from the target (`y`).
6. Exclude the customer ID from model inputs.
7. Split the data into training and test sets using an 80/20 stratified split.
8. Build a preprocessing pipeline using median imputation and scaling for numeric features, and most-frequent imputation and one-hot encoding for categorical features.
9. Train the three models.
10. Compare model performance using accuracy, precision, recall, macro-F1, and ROC-AUC.
11. Generate confusion matrices and error-analysis results.
12. Generate churn probabilities, risk categories, and suggested retention actions.
13. Run validation tests.

## 6. Evaluation
The notebook creates:
- `model_comparison_results.csv`
- `model_error_analysis.csv`
- Confusion-matrix visualizations
- `test_case_report.csv`

Use the actual values in the executed notebook and saved CSV files in the final report. In the recorded run, Gradient Boosting had:
- Accuracy: 0.8062
- Precision: 0.6735
- Recall: 0.5241
- Macro-F1: 0.7313
- ROC-AUC: 0.8432

These scores are specific to this run and test split; they are not a guarantee of performance on future customers.

## 7. Retention Intelligence
The project uses predicted churn probabilities to assign Low, Medium, or High risk categories and attach suggested actions. The thresholds and actions are illustrative and have not been validated through real customer-retention experiments. Predictions are estimates, not guarantees.

## 8. Generated Files
The notebook is intended to generate:
- `model_comparison_results.csv`
- `model_error_analysis.csv`
- `retention_intelligence_results.csv`
- `test_case_report.csv`
- `architecture_diagram.png`

## 9. Requirements
Main libraries:
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

A `requirements.txt` file is included separately. For exact reproducibility, record the package versions from the environment used to run the notebook.

## 10. How to Run
1. Place the dataset CSV in the same folder as `Churn_Prediction.ipynb`.
2. Open the notebook in Jupyter.
3. Run the cells in order, from data loading through validation.
4. Confirm that the output CSV files and architecture diagram are generated.
5. Review the model comparison and test report before using the results in a final report.

## 11. Limitations
- The dataset represents a particular telecommunications customer population and may not generalize to other businesses.
- The model's predictions depend on the selected features, data quality, and classification threshold.
- False negatives can represent customers who churn but are not flagged; false positives can lead to unnecessary retention outreach.
- Risk thresholds and recommended actions need business validation before real-world use.
- This notebook implements churn classification and retention suggestions. If an assessment separately requires a question-answering or retrieval component, that requirement must be implemented and documented separately.
