import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

np.random.seed(42)

def main():
    print("================================================================================")
    print("                    PART 3 — ADVANCED MODELING AND PIPELINES                    ")
    print("================================================================================")
    
    # Load cleaned data (check parent or current folder)
    dataset_path = "c:/Users/intel/Desktop/signbridge/cleaned_data.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "cleaned_data.csv"
        if not os.path.exists(dataset_path):
            dataset_path = "../cleaned_data.csv"
            
    df = pd.read_csv(dataset_path)
    print(f"Loaded cleaned_data.csv. Shape: {df.shape}")
    
    # Setup Target & Encoding
    y_reg = df['Salary']
    y_clf = (y_reg > y_reg.median()).astype(int)
    X = df.drop(columns=['Salary'])
    X_encoded = pd.get_dummies(X, columns=['Department', 'Gender'], drop_first=True, dtype=float)
    
    X_train, X_test, _, _, y_clf_train, y_clf_test = train_test_split(
        X_encoded, y_clf, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Unconstrained Decision Tree
    dt_un = DecisionTreeClassifier(random_state=42)
    dt_un.fit(X_train_scaled, y_clf_train)
    print("Unconstrained Decision Tree:")
    print(f"  Train Acc: {dt_un.score(X_train_scaled, y_clf_train):.4f} | Test Acc: {dt_un.score(X_test_scaled, y_clf_test):.4f}")
    
    # 2. Controlled Decision Tree
    dt_con = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
    dt_con.fit(X_train_scaled, y_clf_train)
    print("\nControlled Decision Tree (max_depth=5, min_samples_split=20):")
    print(f"  Train Acc: {dt_con.score(X_train_scaled, y_clf_train):.4f} | Test Acc: {dt_con.score(X_test_scaled, y_clf_test):.4f}")
    
    # 3. Gini vs Entropy
    dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42).fit(X_train_scaled, y_clf_train)
    dt_ent = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42).fit(X_train_scaled, y_clf_train)
    print("\nCriterion Comparison (depth=5):")
    print(f"  Gini Test Acc: {dt_gini.score(X_test_scaled, y_clf_test):.4f} | Entropy Test Acc: {dt_ent.score(X_test_scaled, y_clf_test):.4f}")
    
    # 4. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_clf_train)
    rf_test_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
    print(f"\nRandom Forest (estimators=100, depth=10) Test AUC: {rf_test_auc:.4f}")
    
    # Feature importances
    rf_importances = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': rf.feature_importances_})
    rf_importances = rf_importances.sort_values(by='Importance', ascending=False)
    print("\nRandom Forest Feature Importances:")
    print(rf_importances)
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X_train_scaled, y_clf_train)
    gb_test_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
    print(f"\nGradient Boosting Test AUC: {gb_test_auc:.4f}")
    
    # Feature ablation study
    lowest_5_features = rf_importances['Feature'].tail(5).tolist()
    X_train_red = X_train.drop(columns=lowest_5_features)
    X_test_red = X_test.drop(columns=lowest_5_features)
    
    scaler_red = StandardScaler()
    X_train_red_scaled = scaler_red.fit_transform(X_train_red)
    X_test_red_scaled = scaler_red.transform(X_test_red)
    
    rf_red = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_train_red_scaled, y_clf_train)
    red_auc = roc_auc_score(y_clf_test, rf_red.predict_proba(X_test_red_scaled)[:, 1])
    print(f"\nFeature Ablation (Removed 5 lowest importances):")
    print(f"  Full model AUC: {rf_test_auc:.4f} | Ablated model AUC: {red_auc:.4f}")
    
    # 5. Cross-validated comparison
    print("\n5-Fold Stratified CV (ROC-AUC):")
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.linear_model import LogisticRegression
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Controlled Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }
    for name, model in models.items():
        pipe = make_pipeline(StandardScaler(), model)
        scores = cross_val_score(pipe, X_encoded, y_clf, cv=cv_strat, scoring='roc_auc', n_jobs=-1)
        print(f"  {name} -> Mean CV AUC: {scores.mean():.4f} | Std: {scores.std():.4f}")
        
    # 6. Hyperparameter tuning with GridSearchCV
    param_grid = {
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [5, 10, None],
        'randomforestclassifier__min_samples_leaf': [1, 5]
    }
    grid_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(random_state=42))
    grid = GridSearchCV(grid_pipe, param_grid, cv=cv_strat, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_clf_train)
    print(f"\nGridSearchCV Best Params: {grid.best_params_}")
    print(f"GridSearchCV Best CV Score: {grid.best_score_:.4f}")
    
    # Save best pipeline to workspace
    best_pipeline = grid.best_estimator_
    joblib.dump(best_pipeline, 'c:/Users/intel/Desktop/signbridge/best_model.pkl')
    # Also save to current directory
    joblib.dump(best_pipeline, 'best_model.pkl')
    print("Serialized best estimator pipeline to 'best_model.pkl'.")
    
    # 6a. Learning curves
    print("\nManual Learning Curve Table:")
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    for f in fractions:
        size = int(f * len(X_train))
        X_sub, y_sub = X_train.iloc[:size], y_clf_train.iloc[:size]
        sub_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(
            n_estimators=grid.best_params_['randomforestclassifier__n_estimators'],
            max_depth=grid.best_params_['randomforestclassifier__max_depth'],
            min_samples_leaf=grid.best_params_['randomforestclassifier__min_samples_leaf'],
            random_state=42
        ))
        sub_pipe.fit(X_sub, y_sub)
        try:
            tr_auc = roc_auc_score(y_sub, sub_pipe.predict_proba(X_sub)[:, 1])
        except:
            tr_auc = 1.0
        te_auc = roc_auc_score(y_clf_test, sub_pipe.predict_proba(X_test)[:, 1])
        print(f"  Fraction: {f*100:.0f}% | Size: {size} | Train AUC: {tr_auc:.4f} | Test AUC: {te_auc:.4f}")
        
    print("================================================================================")
    print("                    PART 3 COMPLETE                                             ")
    print("================================================================================")

if __name__ == "__main__":
    main()
