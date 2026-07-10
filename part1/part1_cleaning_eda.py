import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

def main():
    print("================================================================================")
    print("                    PART 1 — DATA CLEANING AND EXPLORATORY ANALYSIS             ")
    print("================================================================================")
    
    # 1. Load the dataset (from parent directory or current directory)
    dataset_path = "c:/Users/intel/Desktop/signbridge/sample_dataset.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "sample_dataset.csv"
        
    df = pd.read_csv(dataset_path)
    print("\n[Task 1.1] First 5 rows of raw dataset:")
    print(df.head())
    print(f"\n[Task 1.2] DataFrame Shape: {df.shape}")
    print("\n[Task 1.3] Column Dtypes:")
    print(df.dtypes)
    
    # Save raw shapes and counts for later comparison
    raw_shape = df.shape
    raw_null_count = df.isnull().sum()
    raw_null_pct = (raw_null_count / raw_shape[0]) * 100
    
    # 2. Null value analysis
    print("\n[Task 2] Null Value Analysis:")
    null_table = pd.DataFrame({
        'Null Count': raw_null_count,
        'Null Percentage (%)': raw_null_pct
    })
    print(null_table)
    
    exceed_20_null = null_table[null_table['Null Percentage (%)'] > 20]
    if exceed_20_null.empty:
        print("\nNo columns exceed a 20% null rate.")
    else:
        print("\nColumns exceeding 20% null rate:")
        print(exceed_20_null)
        
    # Fill Age and Salary with their median
    age_median = df['Age'].median()
    salary_median = df['Salary'].median()
    df['Age'] = df['Age'].fillna(age_median)
    df['Salary'] = df['Salary'].fillna(salary_median)
    print(f"\nImputed Age with median ({age_median}) and Salary with median ({salary_median}).")
    
    # 3. Duplicate detection and removal
    dup_count = df.duplicated().sum()
    print(f"\n[Task 3] Duplicate Count: {dup_count}")
    df_clean = df.drop_duplicates()
    rows_removed = df.shape[0] - df_clean.shape[0]
    print(f"Removed {rows_removed} duplicate rows. New shape: {df_clean.shape}")
    
    # Check if duplicate removal changed any column's null percentage
    clean_null_pct = (df_clean.isnull().sum() / df_clean.shape[0]) * 100
    print("\nNull Percentage after Duplicate Removal:")
    clean_null_table = pd.DataFrame({
        'Raw Null %': raw_null_pct,
        'Clean Null %': clean_null_pct
    })
    print(clean_null_table)
    
    df = df_clean.copy()
    
    # 4. Data type correction
    mem_before = df.memory_usage(deep=True).sum()
    print(f"\n[Task 4] Deep memory usage before dtype correction: {mem_before / 1024:.2f} KB")
    
    df['Age'] = df['Age'].astype('int64')
    df['Department'] = df['Department'].astype('category')
    df['Gender'] = df['Gender'].astype('category')
    
    mem_after = df.memory_usage(deep=True).sum()
    print(f"Deep memory usage after dtype correction: {mem_after / 1024:.2f} KB")
    print(f"Memory savings: {((mem_before - mem_after) / mem_before) * 100:.2f}%")
    print("New Dtypes:")
    print(df.dtypes)
    
    # 5. Descriptive statistics and skewness
    print("\n[Task 5] Descriptive Statistics for Numeric Columns:")
    print(df.describe())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewness = {}
    print("\nColumn Skewness:")
    for col in numeric_cols:
        skew = df[col].skew()
        skewness[col] = skew
        print(f"  {col}: {skew:.4f}")
        
    highest_skew_col = max(skewness, key=lambda k: abs(skewness[k]))
    print(f"\nColumn with highest absolute skewness: '{highest_skew_col}' (Skewness: {skewness[highest_skew_col]:.4f})")
    
    # 6. Outlier detection with IQR
    print("\n[Task 6] Outlier Detection with IQR:")
    for col in ['Salary', 'Experience']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"  {col} -> Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"    Outlier Count: {outliers.shape[0]} rows out of {df.shape[0]}")
        
    # 7. Visualizations
    print("\n[Task 7] Generating and saving plots...")
    # Plot 1: Line plot
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Salary'], color='#4F46E5', linewidth=1.5, alpha=0.8)
    plt.title('Line Plot of Salary by Row Index', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Row Index', fontsize=11)
    plt.ylabel('Salary', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('line_plot.png', dpi=300)
    plt.close()
    
    # Plot 2: Bar chart
    plt.figure(figsize=(8, 5))
    mean_salary_by_dept = df.groupby('Department', observed=False)['Salary'].mean()
    colors = ['#4F46E5', '#06B6D4', '#10B981', '#F59E0B', '#EF4444']
    plt.bar(mean_salary_by_dept.index, mean_salary_by_dept.values, color=colors, edgecolor='none', width=0.6)
    plt.title('Mean Salary by Department', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Department', fontsize=11)
    plt.ylabel('Mean Salary', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('bar_chart.png', dpi=300)
    plt.close()
    
    # Plot 3: Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df[highest_skew_col], bins=20, kde=True, color='#06B6D4', edgecolor='w', alpha=0.8)
    plt.title(f'Histogram of Most Skewed Column: {highest_skew_col}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(highest_skew_col, fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('histogram.png', dpi=300)
    plt.close()
    
    # Plot 4: Scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Experience', y='Salary', data=df, color='#10B981', alpha=0.7, edgecolor='w', s=60)
    plt.title('Scatter Plot: Salary vs Experience', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Experience (Years)', fontsize=11)
    plt.ylabel('Salary', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('scatter_plot.png', dpi=300)
    plt.close()
    
    # Plot 5: Box plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Department', y='PerformanceScore', data=df, palette='Set2', hue='Department', legend=False)
    plt.title('Performance Score by Department', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Department', fontsize=11)
    plt.ylabel('Performance Score', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('box_plot.png', dpi=300)
    plt.close()
    
    # 8. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    df_numeric = df.select_dtypes(include=[np.number])
    corr_matrix_pearson = df_numeric.corr(method='pearson')
    sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', fmt=".4f", cbar=True, square=True,
                annot_kws={"size": 10}, linewidths=0.5)
    plt.title('Pearson Correlation Heat Map', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)
    plt.close()
    print("All 5 plots and correlation heatmap saved successfully.")
    
    # Spearman rank correlation comparison
    corr_matrix_spearman = df_numeric.corr(method='spearman')
    diff_matrix = (corr_matrix_spearman - corr_matrix_pearson).abs()
    diff_lower = diff_matrix.where(np.tril(np.ones(diff_matrix.shape), k=-1).astype(bool))
    diff_lower_unstacked = diff_lower.unstack().dropna().sort_values(ascending=False)
    
    print("\nSpearman-Pearson Correlation Differences:")
    diff_table = pd.DataFrame(columns=['Pair', 'Spearman', 'Pearson', 'Difference'])
    for idx, (pair, diff_val) in enumerate(diff_lower_unstacked.items()):
        c1, c2 = pair
        diff_table.loc[idx] = [f"{c1} - {c2}", f"{corr_matrix_spearman.loc[c1,c2]:.4f}", f"{corr_matrix_pearson.loc[c1,c2]:.4f}", f"{diff_val:.4f}"]
    print(diff_table.head(5))
    
    # Grouped Aggregation
    print("\nGrouped Aggregation (Department, Salary):")
    grouped_agg = df.groupby('Department', observed=False)['Salary'].agg(['mean', 'std', 'count'])
    print(grouped_agg)
    
    # Save cleaned data
    df.to_csv("c:/Users/intel/Desktop/signbridge/cleaned_data.csv", index=False)
    # Also save to part1 folder
    df.to_csv("cleaned_data.csv", index=False)
    print("\nSaved cleaned dataset 'cleaned_data.csv'.")
    print("================================================================================")
    print("                    PART 1 EXECUTION COMPLETE                                   ")
    print("================================================================================")

if __name__ == "__main__":
    main()
