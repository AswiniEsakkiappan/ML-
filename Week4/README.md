# Task 4 – Household Energy Consumption Prediction (Polynomial Regression)

## Overview
This project predicts the **daily energy consumption (kWh)** of a household using **Polynomial Regression (degree 2)**. The model uses household size, average temperature, and peak-hours usage as inputs and is evaluated with standard regression metrics.

## Dataset
**File:** `household_energy_consumption.csv`  
**Size:** 55,139 rows × 7 columns (55,138 valid rows after cleaning)

| Column | Description |
|---|---|
| `Household_ID` | Unique household identifier (e.g., H00001) |
| `Date` | Date of the reading |
| `Energy_Consumption_kWh` | Daily energy consumed (**target variable**) |
| `Household_Size` | Number of people in the household (1–6) |
| `Avg_Temperature_C` | Average daily temperature in °C (10–25) |
| `Has_AC` | Whether the household has an air conditioner (Yes/No) |
| `Peak_Hours_Usage_kWh` | Energy used during peak hours |

## Project Workflow
1. **Import libraries** – NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
2. **Load data** – read the CSV into a DataFrame
3. **Explore data** – `head()`, `tail()`, `shape`, `info()`, `describe()`, `dtypes`
4. **Clean data** – one incomplete row (all-NaN values) was found and removed with `dropna()`
5. **Select features and target**
   - Features (X): `Household_Size`, `Avg_Temperature_C`, `Peak_Hours_Usage_kWh`
   - Target (y): `Energy_Consumption_kWh`
6. **Train/test split** – 80% training, 20% testing (`random_state=42`)
7. **Feature transformation** – `PolynomialFeatures(degree=2)` fitted on the training set and applied to both sets
8. **Model training** – `LinearRegression` fitted on the polynomial features
9. **Prediction and evaluation** on the test set
10. **Visualization** – scatter plot of actual vs predicted values

## Results

| Metric | Value |
|---|---|
| MAE | 0.5734 |
| MSE | 0.5305 |
| RMSE | 0.7284 |
| R² | 0.9827 |

The model explains roughly **98.3%** of the variance in energy consumption, with an average absolute error of about **0.57 kWh**. The actual-vs-predicted scatter plot shows points close to the diagonal, indicating a strong fit.

**Sample predictions:**

| Actual (kWh) | Predicted (kWh) |
|---|---|
| 11.6 | 11.80 |
| 14.1 | 13.76 |
| 6.9 | 7.64 |
| 4.7 | 4.41 |
| 7.4 | 6.89 |

## Requirements
- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run
1. Place `household_energy_consumption.csv` in your working directory.
2. Update the file path in the notebook. It currently reads `/content/household_energy_consumption.csv` (Google Colab path), so change it to `household_energy_consumption.csv` for local use.
3. Open `Task4_ML.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
4. Run all cells in order.

## Files
```
├── Task4_ML.ipynb                    # Main notebook
├── household_energy_consumption.csv  # Dataset
└── README.md                         # Project documentation
```

## Possible Improvements
- Include the `Has_AC` feature (encode as 0/1) and date-based features such as day of week.
- Compare against plain linear regression and other polynomial degrees to confirm degree 2 is optimal.
- Use cross-validation for more robust evaluation.
- Split by household or by time to avoid leakage, since the same household appears on many rows.
- Use the imported Seaborn for exploratory plots (distributions, correlation heatmap).
