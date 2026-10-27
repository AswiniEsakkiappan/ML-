# Phishing Website Detection using Linear SVM (Week 8)

This project builds a machine learning model that classifies websites as **Phishing** or **Legitimate** based on numeric features extracted from their URLs and pages. It uses a **Linear Support Vector Machine (LinearSVC)** trained on the PhiUSIIL Phishing URL dataset.

## Contents

| File | Description |
| --- | --- |
| `week8.ipynb` | Jupyter/Colab notebook with data loading, preprocessing, training and evaluation |
| `PhiUSIIL_Phishing_URL_Dataset.csv` | Dataset used by the notebook (not included; see below) |

## Dataset

The notebook uses the **PhiUSIIL Phishing URL Dataset**, where each row is a website described by URL and page-based features.

- Target column: `label`
- Non-numeric columns dropped: `URL`, `Domain`, `TLD`, `Title`
- Only numeric features are used for training

In the notebook, the target is encoded as `y = (label == 0)`, so **1 = Phishing** and **0 = Legitimate**.

> The notebook reads the CSV from `/content/` (Google Colab). If you run it elsewhere, update the file path in the `pd.read_csv(...)` call.

## Workflow

1. **Load data**: read the CSV with pandas and inspect shape, columns and class distribution.
2. **Preprocess**:
   - Drop rows with a missing label
   - Remove text columns (`URL`, `Domain`, `TLD`, `Title`)
   - Keep numeric features only
   - Replace `inf` / `-inf` with `NaN`, then fill missing values with the column median
3. **Split**: 80% training / 20% testing, `random_state=42`, stratified by class.
4. **Train**: a scikit-learn pipeline of `StandardScaler` followed by `LinearSVC` (`C=1.0`, `class_weight="balanced"`, `max_iter=5000`, `random_state=42`).
5. **Evaluate** on the test set:
   - Confusion matrix (plotted with `ConfusionMatrixDisplay`)
   - Precision, Recall, F1-score and ROC AUC
   - Full classification report
   - ROC curve

## Requirements

- Python 3.8+
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

Install them with:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## How to Run

1. Download the PhiUSIIL Phishing URL dataset CSV.
2. Open `week8.ipynb` in Jupyter Notebook, JupyterLab or Google Colab.
3. Update the CSV path in the data-loading cells if needed.
4. Run all cells from top to bottom.

## Outputs

Running the notebook produces:

- Dataset shape, missing-value count and class distribution
- Train/test sample counts
- A confusion matrix plot
- Precision, Recall, F1-score and ROC AUC values
- A per-class classification report
- A ROC curve for the Linear SVM

## Notes

- `class_weight="balanced"` is used so the model handles any class imbalance.
- Features are standardized before training, which SVMs need to perform well.
- Median imputation keeps the pipeline robust to missing or infinite values.

## Possible Improvements

- Compare against other models (Logistic Regression, Random Forest, Gradient Boosting)
- Tune `C` with cross-validation / grid search
- Inspect feature importance or model coefficients
- Check for features that may leak the label


## output

<img width="581" height="455" alt="image" src="https://github.com/user-attachments/assets/9b63c5f9-f6ca-4916-91ca-ef9d7905fe4a" />

<img width="448" height="455" alt="image" src="https://github.com/user-attachments/assets/3d69a27a-33c1-4e90-970a-5b77daf05341" />


