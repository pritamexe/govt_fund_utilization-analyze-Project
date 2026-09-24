# Indian Government Scheme Fund Utilization Analysis (2021-2025)

**AICTE x IBM Data Analyst Internship Project**  
**Author:** Pritam Mahato  
**Dataset Source:** [Kaggle - Govt Scheme Fund Utilization 2021-2025 India](https://www.kaggle.com/datasets/arjunsinghgangwar/govt-scheme-fund-utilization-2021-2025-india)

---

## Project Overview

This project performs an end-to-end data analytics and machine-learning study on the **Indian Government Scheme Fund Utilization** dataset covering financial years 2021-22 through 2024-25. The analysis traces how funds sanctioned under major centrally-sponsored and central-sector schemes were released, utilized, and translated into physical and beneficiary outcomes across 10 Indian states.

The project was developed as a submission for the **AICTE x IBM Data Analyst Internship** programme.

---

## Problem Statement

Despite large budgetary allocations, fund under-utilization in government welfare schemes remains a persistent challenge. This project investigates the patterns, correlations, and predictors of fund utilization rates and scheme implementation status across states, departments, and schemes in India between 2021 and 2025.

---

## Objectives

1. Profile the dataset and identify data-quality issues.
2. Perform comprehensive exploratory data analysis across financial, physical, and beneficiary dimensions.
3. Engineer meaningful derived features without introducing target leakage.
4. Train and compare classification models to predict scheme implementation status.
5. Extract evidence-based, quantitative insights on fund utilization patterns.
6. Produce a reproducible, submission-ready project with professional documentation.

---

## Dataset Description

| Attribute             | Value                                          |
|-----------------------|------------------------------------------------|
| Dataset Name          | Indian Govt Scheme Fund Utilization 2021-2025  |
| Source                | Kaggle (see URL above)                         |
| File                  | `govt_fund_utilization.csv`                    |
| **Rows (actual)**     | **1,000**                                      |
| **Columns (actual)**  | **21**                                         |
| Financial Years       | 2021-22, 2022-23, 2023-24, 2024-25             |
| States                | 10                                             |
| Schemes               | 11                                             |
| Departments           | 5                                              |

### Key Columns

| Column                       | Type    | Description                                      |
|------------------------------|---------|--------------------------------------------------|
| Record_ID                    | str     | Unique record identifier                         |
| Financial_Year               | str     | FY of the record (e.g. 2022-23)                  |
| Date_of_Fund_Release         | date    | Date on which funds were released                |
| State                        | str     | Indian state                                     |
| District                     | str     | District name                                    |
| Department                   | str     | Nodal ministry/department                        |
| Scheme_Name                  | str     | Name of the government scheme                    |
| Scheme_Code                  | str     | Short code for the scheme                        |
| Fund_Source                  | str     | Funding category (Central/Centrally Sponsored)   |
| Implementing_Agency          | str     | Body implementing the scheme                     |
| Sanctioned_Amount_Lakhs      | float   | Total amount sanctioned (Rs. Lakhs)              |
| Fund_Released_Lakhs          | float   | Amount released (Rs. Lakhs)                      |
| Fund_Utilized_Lakhs          | float   | Amount actually utilized (Rs. Lakhs) [60 missing]|
| Utilization_Percent          | float   | Utilized as % of Released [60 missing]           |
| Physical_Target_Units        | int     | Physical target set for the period               |
| Physical_Achievement_Units   | float   | Physical units achieved [25 missing]             |
| Beneficiaries_Target         | int     | Target beneficiary count                         |
| Beneficiaries_Covered        | float   | Actual beneficiaries covered [45 missing]        |
| Status                       | str     | Implementation status (4 categories)             |
| Audit_Remarks                | str     | Audit note or remark [70 missing]                |
| Nodal_Officer_Contact        | str     | Contact details of nodal officer                 |

---

## Technologies Used

| Technology     | Purpose                              |
|----------------|--------------------------------------|
| Python 3.x     | Core programming language            |
| pandas         | Data loading, cleaning, aggregation  |
| numpy          | Numerical operations                 |
| matplotlib     | Visualisation                        |
| seaborn        | Statistical visualisation            |
| scikit-learn   | Machine learning, preprocessing      |
| openpyxl       | Excel output support                 |
| python-docx    | DOCX report generation               |

---

## Project Structure

```
IBM/
|-- govt_fund_utilization.csv                            (source dataset - DO NOT MODIFY)
|-- Pritam_Mahato_Govt_Fund_Utilization_Analysis.py      (main analysis script)
|-- requirements.txt                                     (Python dependencies)
|-- README.md                                            (this file)
|-- Pritam_Mahato_Govt_Fund_Utilization_Project_Report.docx
|-- outputs/
    |-- 01_records_by_fy.png
    |-- 02_yearly_utilization_trend.png
    |-- 03_sanction_release_utilize_by_fy.png
    |-- 04_utilization_pct_distribution.png
    |-- 05_state_utilization.png
    |-- 06_department_utilization.png
    |-- 07_scheme_fund_utilized.png
    |-- 08_beneficiary_target_vs_covered.png
    |-- 09_physical_target_vs_achievement.png
    |-- 10_status_distribution.png
    |-- 11_correlation_heatmap.png
    |-- 12_utilization_by_status.png
    |-- 13_scheme_avg_utilization.png
    |-- 14_confusion_matrix_best_model.png
    |-- 15_model_comparison.png
    |-- 16_feature_importance.png
    |-- cleaned_dataset.csv
    |-- missing_value_summary.csv
    |-- statistical_summary.csv
    |-- state_level_summary.csv
    |-- scheme_level_summary.csv
    |-- model_metrics.csv
    |-- predictions.csv
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

Place `govt_fund_utilization.csv` in the same directory as the Python script, then run:

```bash
python Pritam_Mahato_Govt_Fund_Utilization_Analysis.py
```

To specify a custom CSV path:

```bash
python Pritam_Mahato_Govt_Fund_Utilization_Analysis.py --csv path/to/your_data.csv
```

All outputs are written to the `outputs/` directory (created automatically).

---

## Analysis Workflow

```
CSV Dataset
    |
    v
load_and_validate()        # schema check, load
    |
    v
preprocess()               # strip whitespace, parse dates, handle missing values,
    |                      # engineer derived features
    v
run_eda()                  # 13 charts + aggregated statistics
    |
    v
statistical_summary()      # descriptive stats for all numerical columns
    |
    v
run_ml()                   # 3 classifiers, evaluation, confusion matrix,
    |                      # feature importance
    v
save_outputs()             # cleaned dataset, state/scheme summaries, predictions
```

---

## Data Preprocessing

| Issue                               | Treatment                                     | Justification                                          |
|-------------------------------------|-----------------------------------------------|--------------------------------------------------------|
| Fund_Utilized_Lakhs (60 missing)    | Retained as NaN; excluded from calculations   | Imputation would fabricate financial figures           |
| Utilization_Percent (60 missing)    | Retained as NaN (directly derived from above) | Same reason; replacing NaN would introduce false data  |
| Physical_Achievement_Units (25)     | Retained as NaN                               | Cannot estimate actual physical work done              |
| Beneficiaries_Covered (45)          | Retained as NaN                               | Cannot estimate actual beneficiary coverage            |
| Audit_Remarks (70 missing)          | Filled with "Not Available"                   | Textual placeholder; does not affect quantitative analysis |
| District/State string inconsistency | strip() applied                               | Removes leading/trailing whitespace                    |
| Audit_Remarks duplicate variants    | Normalised with regex whitespace collapse     | Multiple records had values with extra internal spaces |

---

## Exploratory Data Analysis

Thirteen visualisations are generated covering:

- Record distribution by financial year
- Yearly utilization trend (mean and median)
- Sanctioned vs Released vs Utilized funds by FY
- Distribution of utilization percentage
- State-wise average utilization
- Department-wise average utilization
- Scheme-wise total fund utilized
- Beneficiary target vs covered
- Physical target vs achievement
- Implementation status distribution
- Correlation heatmap
- Utilization % by status (box plot)
- Average utilization % by scheme

---

## Machine Learning Methodology

**Task:** Multi-class classification (predict implementation Status)

**Target variable:** `Status` - 4 classes: `Completed`, `Delayed`, `Not Started`, `Ongoing`

**Target justification:** Status is a meaningful operational outcome label. It is not a direct mathematical transformation of any input feature, making it a valid ML target.

**Leakage prevention:** `Fund_Utilized_Lakhs`, `Utilization_Percent`, `Physical_Achievement_Units`, and `Beneficiaries_Covered` are excluded because they encode the downstream outcome of implementation. Using them as features would cause data leakage.

**Features used (13):**

| Feature                  | Type        |
|--------------------------|-------------|
| Financial_Year           | Categorical |
| State                    | Categorical |
| Department               | Categorical |
| Scheme_Name              | Categorical |
| Fund_Source              | Categorical |
| Implementing_Agency      | Categorical |
| Sanctioned_Amount_Lakhs  | Numerical   |
| Fund_Released_Lakhs      | Numerical   |
| Release_Sanction_Ratio   | Engineered  |
| Physical_Target_Units    | Numerical   |
| Beneficiaries_Target     | Numerical   |
| Release_Month            | Date-derived|
| Release_Quarter          | Date-derived|

**Train/Test split:** 80% / 20%, stratified, random_state=42

**Models evaluated:** Logistic Regression, Random Forest, Gradient Boosting

---

## Evaluation Metrics (Actual Calculated Results)

| Model                  | Accuracy | F1-Macro |
|------------------------|----------|----------|
| Logistic Regression    | 0.3250   | 0.2129   |
| Random Forest          | 0.4050   | 0.2588   |
| Gradient Boosting      | 0.3450   | **0.2707** |

**Best model:** Gradient Boosting (F1-Macro = 0.2707)

The relatively moderate F1-Macro scores reflect the fact that implementation status in this dataset is not strongly predictable from pre-implementation features alone, which is itself a finding: contextual, scheme-specific, and administrative factors captured in the dataset have limited predictive power for final status.

---

## Key Findings

1. **Mean fund utilization is 68.3%**, with a median of 82.0%, indicating a left-skewed distribution — many records achieve high utilization but a subset performs very poorly.
2. **6% of fund utilization records (60 rows) are missing**, which may indicate records where utilization reporting had not been completed at data collection time.
3. **Record volume increased year-over-year** from 156 records in 2021-22 to 307 in 2023-24 and 274 in 2024-25, reflecting growing scheme activity.
4. **Utilization_Percent is mathematically derived as Fund_Utilized / Fund_Released x 100** (verified against all 940 non-null rows), confirming internal consistency of the dataset.
5. **Fund_Released > Sanctioned: 0 cases** — no anomalous over-release detected.
6. **Completed status** accounts for 34.9% of records, while **Not Started** accounts for 10.8%, representing a potential pool of underperforming records.
7. **Feature importance analysis** indicates that Sanctioned_Amount_Lakhs, Fund_Released_Lakhs, and Release_Sanction_Ratio are the most influential predictors of status among the available leakage-free features.
8. **District strings** contained leading/trailing whitespace inconsistencies in a subset of records, which were corrected during preprocessing.

---

## Generated Outputs

| File                         | Description                              |
|------------------------------|------------------------------------------|
| `outputs/cleaned_dataset.csv`| Cleaned + feature-enriched dataset       |
| `outputs/statistical_summary.csv` | Descriptive statistics for all numerical columns |
| `outputs/state_level_summary.csv` | State-level aggregated summary        |
| `outputs/scheme_level_summary.csv`| Scheme-level aggregated summary       |
| `outputs/model_metrics.csv`  | Model comparison metrics                 |
| `outputs/predictions.csv`    | Test-set predictions vs actual status    |
| `outputs/missing_value_summary.csv` | Missing value audit                |
| `outputs/*.png`              | 16 analysis visualisations               |

---

## Limitations

- The dataset covers 10 states only and may not be representative of all Indian states.
- The Status variable (ML target) may reflect administrative reporting status rather than true physical completion.
- 6% of fund utilization values are missing; analysis of utilization rates is restricted to the 940 non-null records.
- ML classification performance is moderate, suggesting the available features do not fully capture the factors that determine implementation status.
- The dataset appears to be a structured sample rather than an exhaustive administrative record.

---

## Future Scope

- Integrate district-level socioeconomic indicators to improve ML predictability.
- Apply time-series analysis to track fund utilization trajectories within a scheme across years.
- Build a scheme-level risk-scoring model to flag early indicators of delayed implementation.
- Extend to all 28+ Indian states for broader coverage.
- Include satellite/geospatial data to validate physical achievement claims.

---

## References

1. Dataset: Arjun Singh Gangwar, "Govt Scheme Fund Utilization 2021-2025 India", Kaggle, 2024.
   URL: https://www.kaggle.com/datasets/arjunsinghgangwar/govt-scheme-fund-utilization-2021-2025-india
2. Ministry of Finance, Government of India — Public Financial Management System (PFMS) documentation.
3. NITI Aayog — Annual Scheme Monitoring Reports.
4. scikit-learn Documentation: https://scikit-learn.org/stable/
5. pandas Documentation: https://pandas.pydata.org/docs/
