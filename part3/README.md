# Part 3 — Advanced Modeling and Pipelines

## 1. Decision Trees: Unconstrained vs Controlled
- **Unconstrained Tree**: Train Acc = 100%, Test Acc = 60%. (High train-test gap showing extreme overfitting).
- **Controlled Tree (`max_depth=5`, `min_samples_split=20`)**: Train Acc = 67.50%, Test Acc = 45.00%.
- **Variance/Bias**: Decision trees split nodes recursively to fit the training set noise. Controlling depth reduces variance at the cost of some bias.

## 2. Gini vs Entropy Comparison
- **Gini Test Accuracy**: **46.67%**
- **Entropy Test Accuracy**: **48.33%**

#### Formulas
$$\text{Gini} = 1 - \sum p_i^2 \quad , \quad \text{Entropy} = -\sum p_i \log_2(p_i)$$

- A node with Gini = 0 represents perfect purity (all samples in the node belong to a single class).

## 3. Random Forest and Feature Importances
- **Random Forest**: Train Acc = 100%, Test Acc = 66.67%, Test AUC = **0.6144**.
- **Top 5 Features**:
  1. `Age` (Importance: 28.71%)
  2. `Experience` (Importance: 28.33%)
  3. `PerformanceScore` (Importance: 27.96%)
  4. `Gender_Male` (Importance: 4.59%)
  5. `Department_HR` (Importance: 3.77%)

- **Bagging Concept**: Bagging (bootstrap aggregating) averages the predictions of 100 deep decision trees, each trained on a bootstrap sample of the training set with a random subset of $\sqrt{d}$ features considered at each split. This averaging process reduces variance.
- **Ablation Study**: Removing the 5 lowest-importance features dropped the AUC from **0.6144** to **0.5633** (a loss of 0.0511), indicating that these features contain valuable interactive signal.

## 4. Cross-Validated Model Comparison
Evaluating classifiers using 5-Fold Stratified Cross-Validation (ROC-AUC):
- **Logistic Regression**: Mean = 0.5444 | Std = 0.0264
- **Controlled Decision Tree**: Mean = 0.5225 | Std = 0.0453
- **Random Forest**: Mean = 0.4983 | Std = 0.0626
- **Gradient Boosting**: Mean = 0.5205 | Std = 0.0689

## 5. GridSearchCV Pipeline Tuning
- **Best Parameters**: `{'max_depth': 5, 'min_samples_leaf': 5, 'n_estimators': 50}`
- **Best 5-Fold CV AUC**: **0.4873**
- **Grid Search Size**: Evaluated **90 configurations** (18 combinations × 5 folds).

## 6. Learning Curve Analysis
Using the best GridSearchCV pipeline:

| Training Fraction | Train Size | Training AUC | Test AUC |
| :---: | :---: | :---: | :---: |
| 20% | 48 | 0.8881 | 0.5022 |
| 40% | 96 | 0.9121 | 0.5933 |
| 60% | 144 | 0.8671 | 0.5522 |
| 80% | 192 | 0.8701 | 0.5178 |
| 100% | 240 | 0.8517 | 0.5078 |

- **Conclusion**: The model is **capacity-limited** (plateauing test AUC). Collecting more data of the current feature set is unlikely to help; we need to engineer more predictive features.

## 7. Model Serialization & Verification
The best scikit-learn pipeline was serialized using `joblib.dump(best_pipeline, 'best_model.pkl')`.
Verification code block:
```python
import joblib
loaded_model = joblib.load('best_model.pkl')
predictions = loaded_model.predict(X_test.iloc[:2])
print(predictions) # Output: [0, 0]
```
