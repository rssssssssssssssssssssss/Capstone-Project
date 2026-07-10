# Part 2 — Supervised Machine Learning Model

## 1. Feature and Target Definitions
- **Features (`X`)**: `Age`, `Department`, `Gender`, `Experience`, `PerformanceScore`.
- **Regression Target (`y_reg`)**: `Salary`.
- **Classification Target (`y_clf`)**: Binary classification representing whether Salary is above the median ($Salary > \$54,793.00$).
  - Class 0: 164 samples
  - Class 1: 136 samples

## 2. Preprocessing & Leak-Free Split and Scaling
- **Categorical Encoding**: `Department` and `Gender` (nominal categories) are encoded using one-hot encoding with `drop_first=True` to avoid multicollinearity.
- **Train-Test Split**: 80/20 train/test partition using `random_state=42`.
- **Standard Scaling**: `StandardScaler` is fitted strictly on `X_train` and then used to transform both `X_train` and `X_test`. 
  - **Data Leakage Warning**: Fitting the scaler on the full dataset would leak test set stats (mean/std dev) into the training process, causing overly optimistic results.

## 3. Regression Results
- **OLS Linear Regression**: MSE = **$259,540,723.71$**, $R^2 = -0.0574$.
- **Ridge Regression (alpha=1.0)**: MSE = **$259,473,797.75$**, $R^2 = -0.0571$.
- **Coefficients**:
  - `Department_IT`: **+3033.79** (Largest positive coefficient)
  - `Department_HR`: **+1714.66**
  - `Department_Sales`: **-1401.58** (Largest negative coefficient)
  - A positive coefficient means that a standard deviation increase in that feature is associated with a predicted salary increase by the coefficient's value, holding others constant.
- **Ridge vs OLS**: Ridge achieves slightly better performance due to L2 regularization reducing the variance caused by multicollinear features.

## 4. Classification Results (Logistic Regression)
- **Class Balance**: 55.8% (Class 0) vs 44.2% (Class 1). No class balancing is required as the minority class is above the 35% threshold.
- **Performance (C=1.0)**: Test Accuracy = **47%**, Precision = **40.0%**, Recall = **13.3%**, F1-Score = **20.0%**, ROC-AUC = **0.5800**.
- **ROC Curve**: Plotted and saved to `roc_curve.png`.

### Precision and Recall Formulas
$$\text{Precision} = \frac{TP}{TP + FP} \quad , \quad \text{Recall} = \frac{TP}{TP + FN}$$

- **Domain Importance**: For HR talent management, **Recall** is more important. Missing potential high-earning talents (False Negatives) leads to talent attrition.
- **AUC Explanation**: An AUC of 0.58 indicates that the model is only slightly better than random guessing.

## 5. Decision-Threshold Sensitivity
Varying the decision threshold:

| Threshold | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: |
| 0.30 | 0.5088 | 0.9667 | 0.6667 |
| 0.40 | 0.5833 | 0.7000 | 0.6364 |
| 0.50 | 0.4000 | 0.1333 | 0.2000 |
| 0.60 | 0.6000 | 0.1000 | 0.1714 |
| 0.70 | 0.0000 | 0.0000 | 0.0000 |

- **F1-Maximizing Threshold**: **0.30** (F1 = 0.6667). 
- To optimize for Recall, we lower the threshold to 0.30, boosting Recall to 96.67% at the cost of lowering precision to 50.88%.

## 6. Regularization Experiment (C=1.0 vs C=0.01)
- **Baseline (C=1.0)**: Precision = 40.0%, Recall = 13.3%, AUC = **0.5800**
- **Strong L2 (C=0.01)**: Precision = 50.0%, Recall = 10.0%, AUC = **0.5311**
- **Explanation**: A smaller C value specifies stronger L2 regularization. Stronger L2 caused underfitting here, dropping the AUC from 0.5800 to 0.5311.

## 7. Bootstrap Confidence Interval for AUC Difference
- **Mean AUC Difference ($AUC_{C=1.0} - AUC_{C=0.01}$)**: **$0.0502$**
- **95% Confidence Interval**: **$[0.0011, 0.1062]$**
- **Interpretation**: Because the 95% confidence interval **excludes zero**, the performance advantage of the $C=1.0$ baseline model is statistically significant.
