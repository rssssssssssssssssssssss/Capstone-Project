import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

np.random.seed(42)

def main():
    print("================================================================================")
    print("                    PART 2 — SUPERVISED MACHINE LEARNING MODEL                  ")
    print("================================================================================")
    
    # Load cleaned data (check parent folder or current folder)
    dataset_path = "c:/Users/intel/Desktop/signbridge/cleaned_data.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "cleaned_data.csv"
        if not os.path.exists(dataset_path):
            dataset_path = "../cleaned_data.csv"
            
    df = pd.read_csv(dataset_path)
    print(f"Loaded cleaned_data.csv. Shape: {df.shape}")
    
    # 1. Target definition
    y_reg = df['Salary']
    y_clf = (y_reg > y_reg.median()).astype(int)
    X = df.drop(columns=['Salary'])
    
    # 2. Categorical Encoding (Nominal one-hot encoding, drop first category)
    X_encoded = pd.get_dummies(X, columns=['Department', 'Gender'], drop_first=True, dtype=float)
    
    # 3. Leak-free split & scale
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X_encoded, y_reg, y_clf, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Regression Model — Linear Regression & Ridge
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_reg_train)
    y_pred_reg = lr.predict(X_test_scaled)
    
    mse_lr = mean_squared_error(y_reg_test, y_pred_reg)
    r2_lr = r2_score(y_reg_test, y_pred_reg)
    print("\nOLS Linear Regression:")
    print(f"  MSE: {mse_lr:.4f} | R²: {r2_lr:.4f}")
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_reg_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    print(f"\nRidge Regression (alpha=1.0):")
    print(f"  MSE: {mean_squared_error(y_reg_test, y_pred_ridge):.4f} | R²: {r2_score(y_reg_test, y_pred_ridge):.4f}")
    
    # 5. Classification Model — Logistic Regression
    clf_train_counts = y_clf_train.value_counts(normalize=True)
    use_balanced = (clf_train_counts.min() < 0.35)
    class_weight_param = 'balanced' if use_balanced else None
    
    clf = LogisticRegression(class_weight=class_weight_param, max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_clf_train)
    y_pred_clf = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    print("\nLogistic Regression (C=1.0) Classification Report:")
    print(confusion_matrix(y_clf_test, y_pred_clf))
    print(classification_report(y_clf_test, y_pred_clf))
    
    auc_clf = roc_auc_score(y_clf_test, y_pred_proba)
    print(f"ROC-AUC: {auc_clf:.4f}")
    
    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_clf_test, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#4F46E5', lw=2, label=f'ROC Curve (AUC = {auc_clf:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    plt.close()
    
    # Sensitivity analysis
    print("\nDecision-Threshold Sensitivity Table:")
    thresholds = np.arange(0.30, 0.71, 0.10)
    for t in thresholds:
        preds_t = (y_pred_proba >= t).astype(int)
        p = precision_score(y_clf_test, preds_t, zero_division=0)
        r = recall_score(y_clf_test, preds_t, zero_division=0)
        f1 = f1_score(y_clf_test, preds_t, zero_division=0)
        print(f"  Threshold: {t:.2f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    
    # 6. Regularization experiment
    clf_strong = LogisticRegression(C=0.01, class_weight=class_weight_param, max_iter=1000, random_state=42)
    clf_strong.fit(X_train_scaled, y_clf_train)
    y_pred_proba_strong = clf_strong.predict_proba(X_test_scaled)[:, 1]
    
    print("\nRegularization Experiment:")
    print(f"  C=1.0 AUC: {auc_clf:.4f}")
    print(f"  C=0.01 AUC: {roc_auc_score(y_clf_test, y_pred_proba_strong):.4f}")
    
    # 7. Bootstrap
    n_bootstraps = 500
    auc_diffs = []
    y_test_np = y_clf_test.values
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(y_test_np), size=len(y_test_np), replace=True)
        try:
            diff = roc_auc_score(y_test_np[idx], y_pred_proba[idx]) - roc_auc_score(y_test_np[idx], y_pred_proba_strong[idx])
            auc_diffs.append(diff)
        except ValueError:
            continue
            
    print(f"\nBootstrap Mean AUC Difference (C=1.0 - C=0.01): {np.mean(auc_diffs):.4f}")
    print(f"95% Confidence Interval for AUC difference: [{np.percentile(auc_diffs, 2.5):.4f}, {np.percentile(auc_diffs, 97.5):.4f}]")
    print("================================================================================")
    print("                    PART 2 COMPLETE                                             ")
    print("================================================================================")

if __name__ == "__main__":
    main()
