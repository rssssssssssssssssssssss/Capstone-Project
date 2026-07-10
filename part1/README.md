# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## 1. Dataset Description
The raw dataset contains **305 records** representing employee profiles with 6 columns:
- `Age` (Numeric): Inferred as `float64` due to NaNs.
- `Salary` (Numeric): Inferred as `float64` due to NaNs.
- `Department` (Categorical: HR, Sales, IT, Marketing, Finance).
- `Gender` (Categorical: Male, Female).
- `Experience` (Numeric): Experience in years (`int64`).
- `PerformanceScore` (Numeric): Performance rating (`int64`).

## 2. Null Value Analysis
Initial missing value counts and percentages:
- **`Age`**: 20 missing values (**6.56%**)
- **`Salary`**: 26 missing values (**8.52%**)
- All other columns had 0 missing values.
- **No column exceeded the 20% null threshold.**

### Median Imputation Justification
Missing values in `Age` and `Salary` were filled using the column **median** rather than the mean. 
For variables like `Salary` (with a max of $300k compared to a median of $54.7k), the distribution is heavily skewed. The **mean** is highly sensitive to extreme outliers, while the **median** represents the 50th percentile and is robust to outliers, preventing biased imputations.

## 3. Duplicate Detection and Removal
- **Duplicates Found**: 5 identical rows.
- **Action**: Removed using `df.drop_duplicates()`.
- **Impact**: Reduced shape from 305 to **300 rows**.

## 4. Data Type Correction
- **`Age`**: Cast from `float64` to `int64` post-imputation since age is a discrete count.
- **`Department` & `Gender`**: Cast from string to `category` dtype.
- **Memory Savings**:
  - **Memory before conversion**: **41.13 KB**
  - **Memory after conversion**: **10.46 KB**
  - **Memory savings**: **74.57%**

## 5. Descriptive Statistics & Skewness
Skewness coefficients of the numeric columns:
- `Age`: $-0.1149$
- `Salary`: **$+5.7087$** (highest absolute skewness)
- `Experience`: $-0.0252$
- `PerformanceScore`: $-0.0790$

**`Salary` skewness is $+5.7087$ (extreme positive skew).**
- **Meaning**: The tail on the right side of the distribution is longer/fatter. Most values lie below the mean, with a few high-value outliers stretching the mean upward.
- **Consequence for Mean Imputation**: Imputing with the mean ($55,954.68) instead of the median ($54,793.00) would over-inflate lower salaries, biasing downstream models.

## 6. Outlier Detection with IQR
Using the standard Interquartile Range ($IQR = Q_3 - Q_1$):
- **`Salary` Outliers**: **10 rows** fall outside bounds $[$ \$21,389.62, \$88,008.62 $]$.
- **`Experience` Outliers**: **0 rows** fall outside bounds $[ -15.50, 52.50 ]$.
- **Handling**: Retained for modeling to capture true high-earning patterns, but scaled in Part 2.

## 7. Visualizations Produced
- **`line_plot.png`**: Plot of Salary by row index.
- **`bar_chart.png`**: Mean Salary by Department. IT has the highest average salary (~$62.6k), while Sales has the lowest (~$52.6k).
- **`histogram.png`**: Distribution of the most skewed column (`Salary`) showing a right-skewed tail.
- **`scatter_plot.png`**: Salary vs Experience, showing a flat, weak relationship.
- **`box_plot.png`**: PerformanceScore by Department, showing relatively uniform medians (around 77).
- **`correlation_heatmap.png`**: Pearson correlation heatmap. Highest absolute correlation is between `Experience` and `PerformanceScore` at a very weak **0.0746**.

## 8. Spearman Rank Correlation
Spearman captures non-linear monotonic relationships:
- **Age - Salary**: Spearman is $0.0266$, Pearson is $-0.0427$ (absolute difference of **$0.0693$**).
- **Salary - Experience**: Spearman is $-0.0628$, Pearson is $-0.0156$ (absolute difference of **$0.0472$**).
We rely on **Spearman Rank Correlation** for non-linear feature-selection guidance.

## 9. Grouped Aggregation
- **IT Department**: Highest Mean Salary ($62,627.21) and highest standard dev ($37,356.42).
- **Implications**: The high std dev in IT means department name alone cannot predict salary reliably. The model requires other features to resolve within-group variance.
- **Mean Ratio**: The ratio of the highest mean salary (IT) to the lowest (Sales) is **1.1910** (a 19% variance), indicating that department carries a helpful predictive signal.
