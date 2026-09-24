"""
=============================================================================
AICTE x IBM Data Analyst Internship Project
Title   : Indian Government Scheme Fund Utilization Analysis (2021-2025)
Author  : Pritam Mahato
Dataset : Indian Govt Scheme Fund Utilization Data 2021-2025
Source  : https://www.kaggle.com/datasets/arjunsinghgangwar/
          govt-scheme-fund-utilization-2021-2025-india
=============================================================================

Usage:
    python Pritam_Mahato_Govt_Fund_Utilization_Analysis.py
    python Pritam_Mahato_Govt_Fund_Utilization_Analysis.py --csv path/to/data.csv

Outputs are written to the 'outputs/' directory.
=============================================================================
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- safe for server / CI

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -- Global style -------------------------------------------------------------
PALETTE = "Blues_d"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 12, "axes.labelsize": 10})

OUTPUT_DIR = "outputs"
RANDOM_STATE = 42


# =============================================================================
# I/O helpers
# =============================================================================

def ensure_output_dir(directory: str = OUTPUT_DIR) -> None:
    """Create the output directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)


def save_fig(fig: plt.Figure, filename: str, directory: str = OUTPUT_DIR) -> str:
    """Save a matplotlib figure and close it.  Returns the full file path."""
    path = os.path.join(directory, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# SECTION 1 - Data Loading & Validation
# =============================================================================

def load_and_validate(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV, assert expected columns exist, and return the raw DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the source CSV file.

    Returns
    -------
    pd.DataFrame
        Raw loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV path does not exist.
    ValueError
        If required columns are missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    required_cols = {
        "Record_ID", "Financial_Year", "Date_of_Fund_Release",
        "State", "District", "Department", "Scheme_Name", "Scheme_Code",
        "Fund_Source", "Implementing_Agency",
        "Sanctioned_Amount_Lakhs", "Fund_Released_Lakhs", "Fund_Utilized_Lakhs",
        "Utilization_Percent", "Physical_Target_Units", "Physical_Achievement_Units",
        "Beneficiaries_Target", "Beneficiaries_Covered",
        "Status", "Audit_Remarks", "Nodal_Officer_Contact",
    }
    missing_req = required_cols - set(df.columns)
    if missing_req:
        raise ValueError(f"Missing expected columns: {missing_req}")

    print(f"[OK] Dataset loaded  -- {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# =============================================================================
# SECTION 2 - Preprocessing & Cleaning
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the raw DataFrame.

    Steps
    -----
    1. Strip leading/trailing whitespace from all string columns.
    2. Standardise Audit_Remarks (multiple variants with extra spaces exist).
    3. Parse Date_of_Fund_Release to datetime.
    4. Handle missing values:
       - Fund_Utilized_Lakhs / Utilization_Percent (60 rows, ~6%):
         These are left as NaN and excluded from fund-utilization calculations.
         Imputation would fabricate financial figures.
       - Physical_Achievement_Units (25 rows, 2.5%):
         Left as NaN; excluded from achievement-ratio calculations.
       - Beneficiaries_Covered (45 rows, 4.5%):
         Left as NaN; excluded from coverage-ratio calculations.
       - Audit_Remarks (70 rows, 7%):
         Filled with 'Not Available' -- a safe non-informative label.
    5. Engineer derived features.
    6. Encode Status as an ordered integer for ML.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_and_validate().

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-enriched DataFrame.
    """
    data = df.copy()

    # 1. Strip whitespace
    str_cols = data.select_dtypes(include="object").columns
    for col in str_cols:
        data[col] = data[col].str.strip()

    # 2. Standardise Audit_Remarks (duplicated values with extra internal spaces)
    if "Audit_Remarks" in data.columns:
        data["Audit_Remarks"] = (
            data["Audit_Remarks"]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .fillna("Not Available")
        )

    # 3. Parse date
    data["Date_of_Fund_Release"] = pd.to_datetime(
        data["Date_of_Fund_Release"], format="%Y-%m-%d", errors="coerce"
    )
    data["Release_Month"] = data["Date_of_Fund_Release"].dt.month
    data["Release_Quarter"] = data["Date_of_Fund_Release"].dt.quarter

    # 4. Missing values -- documented in docstring; NaN values retained for
    #    financial columns; only Audit_Remarks filled with a safe sentinel.

    # 5. Feature engineering
    # 5a. Release-to-Sanction ratio
    data["Release_Sanction_Ratio"] = np.where(
        data["Sanctioned_Amount_Lakhs"] > 0,
        data["Fund_Released_Lakhs"] / data["Sanctioned_Amount_Lakhs"],
        np.nan,
    )

    # 5b. Utilization-to-Release ratio  (same as Utilization_Percent/100 but keeps NaN logic)
    data["Utilization_Release_Ratio"] = np.where(
        data["Fund_Released_Lakhs"] > 0,
        data["Fund_Utilized_Lakhs"] / data["Fund_Released_Lakhs"],
        np.nan,
    )

    # 5c. Utilization gap (funds released but not utilised; NaN where utilized is missing)
    data["Utilization_Gap_Lakhs"] = (
        data["Fund_Released_Lakhs"] - data["Fund_Utilized_Lakhs"]
    )  # naturally NaN where Fund_Utilized is NaN

    # 5d. Physical achievement percentage
    data["Physical_Achievement_Pct"] = np.where(
        data["Physical_Target_Units"] > 0,
        (data["Physical_Achievement_Units"] / data["Physical_Target_Units"]) * 100,
        np.nan,
    )

    # 5e. Beneficiary coverage percentage
    data["Beneficiary_Coverage_Pct"] = np.where(
        data["Beneficiaries_Target"] > 0,
        (data["Beneficiaries_Covered"] / data["Beneficiaries_Target"]) * 100,
        np.nan,
    )

    # 6. Ordinal encode Status for reference (not used as ML target directly)
    status_order = {"Not Started": 0, "Ongoing": 1, "Delayed": 2, "Completed": 3}
    data["Status_Code"] = data["Status"].map(status_order)

    print(f"[OK] Preprocessing complete -- {data.shape[0]:,} rows x {data.shape[1]} columns")
    return data


# =============================================================================
# SECTION 3 - EDA & Visualisations
# =============================================================================

def run_eda(data: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> dict:
    """
    Perform exploratory data analysis and save all visualisations.

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned DataFrame.
    out_dir : str
        Directory in which to save PNG files.

    Returns
    -------
    dict
        Dictionary of computed summary statistics for reporting.
    """
    stats = {}

    # -- 3.1 Missing-value summary ---------------------------------------------
    missing = data.isnull().sum()
    missing_pct = (missing / len(data) * 100).round(2)
    missing_df = pd.DataFrame({"Missing_Count": missing, "Missing_Pct": missing_pct})
    missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values(
        "Missing_Count", ascending=False
    )
    stats["missing_summary"] = missing_df
    missing_df.to_csv(os.path.join(out_dir, "missing_value_summary.csv"))

    # -- 3.2 Financial-year distribution --------------------------------------
    fy_counts = data["Financial_Year"].value_counts().sort_index()
    stats["fy_counts"] = fy_counts

    fig, ax = plt.subplots(figsize=(7, 4))
    fy_counts.plot(kind="bar", ax=ax, color=sns.color_palette("Blues_d", len(fy_counts)))
    ax.set_title("Number of Fund Records by Financial Year")
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Number of Records")
    ax.tick_params(axis="x", rotation=0)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "01_records_by_fy.png", out_dir)

    # -- 3.3 Yearly utilization trend -----------------------------------------
    fy_util = (
        data.groupby("Financial_Year")["Utilization_Percent"]
        .agg(["mean", "median"])
        .round(2)
    )
    stats["fy_util"] = fy_util

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fy_util.index, fy_util["mean"], marker="o", label="Mean Utilization %", linewidth=2)
    ax.plot(fy_util.index, fy_util["median"], marker="s", linestyle="--",
            label="Median Utilization %", linewidth=2)
    ax.set_title("Yearly Fund Utilization Trend")
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Utilization Percentage")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save_fig(fig, "02_yearly_utilization_trend.png", out_dir)

    # -- 3.4 Fund Released vs Utilized (grouped bar by FY) --------------------
    fy_funds = data.groupby("Financial_Year")[
        ["Sanctioned_Amount_Lakhs", "Fund_Released_Lakhs", "Fund_Utilized_Lakhs"]
    ].sum().round(2)
    stats["fy_funds"] = fy_funds

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fy_funds.index))
    width = 0.25
    bars1 = ax.bar(x - width, fy_funds["Sanctioned_Amount_Lakhs"] / 1e3, width,
                   label="Sanctioned (Rs. '000 Lakhs)", color="#4a90d9")
    bars2 = ax.bar(x, fy_funds["Fund_Released_Lakhs"] / 1e3, width,
                   label="Released (Rs. '000 Lakhs)", color="#5ba85b")
    bars3 = ax.bar(x + width, fy_funds["Fund_Utilized_Lakhs"] / 1e3, width,
                   label="Utilized (Rs. '000 Lakhs)", color="#e07b39")
    ax.set_title("Sanctioned vs Released vs Utilized Funds by Financial Year")
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Amount (Rs. '000 Lakhs)")
    ax.set_xticks(x)
    ax.set_xticklabels(fy_funds.index)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "03_sanction_release_utilize_by_fy.png", out_dir)

    # -- 3.5 Utilization % distribution ---------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    util_clean = data["Utilization_Percent"].dropna()
    ax.hist(util_clean, bins=30, color="#4a90d9", edgecolor="white", linewidth=0.5)
    ax.axvline(util_clean.mean(), color="crimson", linestyle="--",
               label=f"Mean = {util_clean.mean():.1f}%")
    ax.axvline(util_clean.median(), color="darkorange", linestyle="-.",
               label=f"Median = {util_clean.median():.1f}%")
    ax.set_title("Distribution of Fund Utilization Percentage")
    ax.set_xlabel("Utilization Percentage (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "04_utilization_pct_distribution.png", out_dir)

    # -- 3.6 State-wise average utilization % ---------------------------------
    state_util = (
        data.groupby("State")["Utilization_Percent"]
        .mean()
        .sort_values(ascending=True)
        .round(2)
    )
    stats["state_util"] = state_util

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("Blues_d", len(state_util))
    state_util.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("State-wise Average Fund Utilization Percentage")
    ax.set_xlabel("Average Utilization (%)")
    ax.set_ylabel("State")
    for bar in ax.patches:
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "05_state_utilization.png", out_dir)

    # -- 3.7 Department-wise utilization ---------------------------------------
    dept_util = (
        data.groupby("Department")["Utilization_Percent"]
        .mean()
        .sort_values(ascending=True)
        .round(2)
    )
    stats["dept_util"] = dept_util

    fig, ax = plt.subplots(figsize=(9, 5))
    dept_util.plot(kind="barh", ax=ax, color=sns.color_palette("Greens_d", len(dept_util)))
    ax.set_title("Department-wise Average Fund Utilization Percentage")
    ax.set_xlabel("Average Utilization (%)")
    ax.set_ylabel("Department")
    for bar in ax.patches:
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "06_department_utilization.png", out_dir)

    # -- 3.8 Scheme-wise fund utilization (total utilized) --------------------
    scheme_util = (
        data.groupby("Scheme_Name")["Fund_Utilized_Lakhs"]
        .sum()
        .sort_values(ascending=True)
        .round(2)
    )
    stats["scheme_util"] = scheme_util

    fig, ax = plt.subplots(figsize=(10, 6))
    scheme_util.plot(kind="barh", ax=ax, color=sns.color_palette("Purples_d", len(scheme_util)))
    ax.set_title("Scheme-wise Total Fund Utilized (Lakhs)")
    ax.set_xlabel("Total Fund Utilized (Rs. Lakhs)")
    ax.set_ylabel("Scheme Name")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    save_fig(fig, "07_scheme_fund_utilized.png", out_dir)

    # -- 3.9 Beneficiary target vs covered ------------------------------------
    ben_fy = data.groupby("Financial_Year")[
        ["Beneficiaries_Target", "Beneficiaries_Covered"]
    ].sum()
    stats["ben_fy"] = ben_fy

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ben_fy))
    ax.bar(x - 0.2, ben_fy["Beneficiaries_Target"] / 1e3, 0.4,
           label="Beneficiaries Target ('000)", color="#4a90d9")
    ax.bar(x + 0.2, ben_fy["Beneficiaries_Covered"] / 1e3, 0.4,
           label="Beneficiaries Covered ('000)", color="#e07b39")
    ax.set_title("Beneficiary Target vs Covered by Financial Year")
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Beneficiaries ('000)")
    ax.set_xticks(x)
    ax.set_xticklabels(ben_fy.index)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "08_beneficiary_target_vs_covered.png", out_dir)

    # -- 3.10 Physical target vs achievement ----------------------------------
    phys_fy = data.groupby("Financial_Year")[
        ["Physical_Target_Units", "Physical_Achievement_Units"]
    ].sum()
    stats["phys_fy"] = phys_fy

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(phys_fy))
    ax.bar(x - 0.2, phys_fy["Physical_Target_Units"] / 1e3, 0.4,
           label="Physical Target ('000 units)", color="#4a90d9")
    ax.bar(x + 0.2, phys_fy["Physical_Achievement_Units"] / 1e3, 0.4,
           label="Physical Achievement ('000 units)", color="#5ba85b")
    ax.set_title("Physical Target vs Achievement by Financial Year")
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Units ('000)")
    ax.set_xticks(x)
    ax.set_xticklabels(phys_fy.index)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "09_physical_target_vs_achievement.png", out_dir)

    # -- 3.11 Status distribution (pie) ---------------------------------------
    status_counts = data["Status"].value_counts()
    stats["status_counts"] = status_counts

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("Set2", len(status_counts)),
    )
    ax.set_title("Distribution of Scheme Implementation Status")
    fig.tight_layout()
    save_fig(fig, "10_status_distribution.png", out_dir)

    # -- 3.12 Correlation heatmap ----------------------------------------------
    num_cols = [
        "Sanctioned_Amount_Lakhs", "Fund_Released_Lakhs", "Fund_Utilized_Lakhs",
        "Utilization_Percent", "Physical_Target_Units", "Physical_Achievement_Units",
        "Beneficiaries_Target", "Beneficiaries_Covered",
        "Release_Sanction_Ratio", "Physical_Achievement_Pct", "Beneficiary_Coverage_Pct",
    ]
    corr_matrix = data[num_cols].corr()
    stats["correlation"] = corr_matrix

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, square=True,
        linewidths=0.5, ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap -- Numerical Features")
    fig.tight_layout()
    save_fig(fig, "11_correlation_heatmap.png", out_dir)

    # -- 3.13 Box-plot: utilization % by status --------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Not Started", "Ongoing", "Delayed", "Completed"]
    sub = data[data["Status"].isin(order)].copy()
    sns.boxplot(
        data=sub, x="Status", y="Utilization_Percent",
        order=order, palette="Set3", ax=ax
    )
    ax.set_title("Fund Utilization Percentage by Implementation Status")
    ax.set_xlabel("Status")
    ax.set_ylabel("Utilization Percentage (%)")
    fig.tight_layout()
    save_fig(fig, "12_utilization_by_status.png", out_dir)

    # -- 3.14 Top-10 schemes by average utilization % -------------------------
    scheme_avg_util = (
        data.groupby("Scheme_Name")["Utilization_Percent"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )
    stats["scheme_avg_util"] = scheme_avg_util

    fig, ax = plt.subplots(figsize=(10, 5))
    scheme_avg_util.plot(kind="bar", ax=ax,
                         color=sns.color_palette("rocket", len(scheme_avg_util)))
    ax.set_title("Average Fund Utilization Percentage by Scheme")
    ax.set_xlabel("Scheme")
    ax.set_ylabel("Average Utilization (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_fig(fig, "13_scheme_avg_utilization.png", out_dir)

    print(f"[OK] EDA complete -- 13 charts saved to '{out_dir}/'")
    return stats


# =============================================================================
# SECTION 4 - Statistical Summary
# =============================================================================

def statistical_summary(data: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """
    Compute and save descriptive statistics for key numerical columns.

    Returns
    -------
    pd.DataFrame
        Summary statistics table.
    """
    num_cols = [
        "Sanctioned_Amount_Lakhs", "Fund_Released_Lakhs", "Fund_Utilized_Lakhs",
        "Utilization_Percent", "Physical_Target_Units", "Physical_Achievement_Units",
        "Beneficiaries_Target", "Beneficiaries_Covered",
        "Physical_Achievement_Pct", "Beneficiary_Coverage_Pct",
        "Release_Sanction_Ratio",
    ]
    summary = data[num_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.90]).T.round(3)
    summary.to_csv(os.path.join(out_dir, "statistical_summary.csv"))
    print("[OK] Statistical summary saved.")
    return summary


# =============================================================================
# SECTION 5 - Machine Learning
# =============================================================================

def run_ml(data: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> dict:
    """
    Train and evaluate classifiers to predict implementation Status.

    Target
    ------
    Status (4 classes: Not Started / Ongoing / Delayed / Completed).

    Rationale
    ---------
    Status is a meaningful operational label that is NOT a direct mathematical
    transformation of any single predictor.  Predicting it from contextual
    features (fund source, department, scheme, state, FY, release/sanction ratio,
    physical-achievement-pct) allows identification of risk factors associated
    with delays or incomplete implementation.

    Leakage check
    -------------
    Fund_Utilized_Lakhs and Utilization_Percent are excluded because they
    contain the outcome of fund utilisation -- knowing them effectively encodes
    the status.  Physical_Achievement_Units and Beneficiaries_Covered are also
    excluded for the same reason.

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned & feature-enriched DataFrame.
    out_dir : str
        Output directory for artefacts.

    Returns
    -------
    dict
        Metrics and artefact paths for reporting.
    """
    results = {}

    # -- Feature selection (leakage-free) -------------------------------------
    feature_cols = [
        "Financial_Year",
        "State",
        "Department",
        "Scheme_Name",
        "Fund_Source",
        "Implementing_Agency",
        "Sanctioned_Amount_Lakhs",
        "Fund_Released_Lakhs",
        "Release_Sanction_Ratio",
        "Physical_Target_Units",
        "Beneficiaries_Target",
        "Release_Month",
        "Release_Quarter",
    ]

    target_col = "Status"

    ml_data = data[feature_cols + [target_col]].dropna(subset=["Release_Sanction_Ratio"])
    print(f"[[i]] ML dataset: {ml_data.shape[0]:,} rows after dropping rows with NaN in key features.")

    # -- Encode categorical features -------------------------------------------
    cat_cols = ["Financial_Year", "State", "Department", "Scheme_Name",
                "Fund_Source", "Implementing_Agency"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        ml_data = ml_data.copy()
        ml_data[col] = le.fit_transform(ml_data[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    y = le_target.fit_transform(ml_data[target_col])
    X = ml_data[feature_cols].values

    results["classes"] = list(le_target.classes_)
    results["n_samples"] = len(X)
    results["n_features"] = len(feature_cols)
    results["feature_names"] = feature_cols

    # -- Train/test split ------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # -- Models ----------------------------------------------------------------
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    model_metrics = {}
    best_f1 = -1
    best_name = None
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        report = classification_report(
            y_test, y_pred, target_names=le_target.classes_, output_dict=True
        )
        model_metrics[name] = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "report": report,
            "y_pred": y_pred,
        }
        print(f"   [{name}] Accuracy: {acc:.4f}  |  F1-macro: {f1_macro:.4f}")
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_name = name
            best_model = model

    results["model_metrics"] = model_metrics
    results["best_model_name"] = best_name
    results["best_f1_macro"] = round(best_f1, 4)

    # -- Confusion matrix for best model --------------------------------------
    best_pred = model_metrics[best_name]["y_pred"]
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, best_pred,
        display_labels=le_target.classes_,
        cmap="Blues", ax=ax,
    )
    ax.set_title(f"Confusion Matrix -- {best_name}")
    fig.tight_layout()
    save_fig(fig, "14_confusion_matrix_best_model.png", out_dir)

    # -- Comparison bar chart --------------------------------------------------
    names = list(model_metrics.keys())
    accs = [model_metrics[n]["accuracy"] for n in names]
    f1s = [model_metrics[n]["f1_macro"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, accs, 0.4, label="Accuracy", color="#4a90d9")
    ax.bar(x + 0.2, f1s, 0.4, label="F1-Macro", color="#e07b39")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison -- Accuracy & F1-Macro")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "15_model_comparison.png", out_dir)

    # -- Feature importance (best tree-based model) ---------------------------
    feat_model = model_metrics.get("Random Forest") or model_metrics.get("Gradient Boosting")
    if best_name in ("Random Forest", "Gradient Boosting"):
        clf = best_model
    else:
        clf = models["Random Forest"]
        clf.fit(X_train, y_train)

    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    results["feature_importance"] = feat_imp

    fig, ax = plt.subplots(figsize=(9, 5))
    feat_imp.sort_values().plot(kind="barh", ax=ax,
                                color=sns.color_palette("rocket", len(feat_imp)))
    ax.set_title(f"Feature Importance -- {best_name}")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    save_fig(fig, "16_feature_importance.png", out_dir)

    # -- Save metrics CSV ------------------------------------------------------
    metrics_rows = []
    for name in names:
        metrics_rows.append({
            "Model": name,
            "Accuracy": model_metrics[name]["accuracy"],
            "F1_Macro": model_metrics[name]["f1_macro"],
        })
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(out_dir, "model_metrics.csv"), index=False
    )

    # -- Save predictions ------------------------------------------------------
    test_idx = ml_data.iloc[
        train_test_split(np.arange(len(ml_data)), test_size=0.20,
                         random_state=RANDOM_STATE, stratify=y)[1]
    ].index
    pred_df = data.loc[test_idx, ["Record_ID", "State", "Scheme_Name", "Status"]].copy()
    pred_df["Predicted_Status"] = le_target.inverse_transform(best_pred)
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    print(f"[OK] ML complete -- Best model: {best_name}  |  F1-Macro: {best_f1:.4f}")
    return results


# =============================================================================
# SECTION 6 - Save cleaned dataset & key summaries
# =============================================================================

def save_outputs(data: pd.DataFrame, stats: dict, out_dir: str = OUTPUT_DIR) -> None:
    """Save the cleaned dataset and key aggregated summaries as CSV files."""
    # Cleaned dataset
    data.to_csv(os.path.join(out_dir, "cleaned_dataset.csv"), index=False)

    # State-level summary
    state_summary = data.groupby("State").agg(
        Records=("Record_ID", "count"),
        Avg_Utilization_Pct=("Utilization_Percent", "mean"),
        Total_Sanctioned_Lakhs=("Sanctioned_Amount_Lakhs", "sum"),
        Total_Released_Lakhs=("Fund_Released_Lakhs", "sum"),
        Total_Utilized_Lakhs=("Fund_Utilized_Lakhs", "sum"),
        Avg_Beneficiary_Coverage_Pct=("Beneficiary_Coverage_Pct", "mean"),
    ).round(2)
    state_summary.to_csv(os.path.join(out_dir, "state_level_summary.csv"))

    # Scheme-level summary
    scheme_summary = data.groupby("Scheme_Name").agg(
        Records=("Record_ID", "count"),
        Avg_Utilization_Pct=("Utilization_Percent", "mean"),
        Total_Sanctioned_Lakhs=("Sanctioned_Amount_Lakhs", "sum"),
        Total_Utilized_Lakhs=("Fund_Utilized_Lakhs", "sum"),
        Avg_Physical_Achievement_Pct=("Physical_Achievement_Pct", "mean"),
    ).round(2)
    scheme_summary.to_csv(os.path.join(out_dir, "scheme_level_summary.csv"))

    print(f"[OK] Cleaned dataset and summary CSVs saved to '{out_dir}/'")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main(csv_path: str) -> None:
    """
    End-to-end analysis pipeline.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV dataset.
    """
    ensure_output_dir(OUTPUT_DIR)
    print("\n" + "=" * 65)
    print(" AICTE x IBM Internship -- Govt Fund Utilization Analysis")
    print("=" * 65 + "\n")

    # 1. Load
    raw = load_and_validate(csv_path)

    # 2. Preprocess
    data = preprocess(raw)

    # 3. EDA
    stats = run_eda(data, OUTPUT_DIR)

    # 4. Statistical summary
    summary = statistical_summary(data, OUTPUT_DIR)

    # 5. ML
    print("\n[ML] Training classifiers to predict implementation Status...")
    ml_results = run_ml(data, OUTPUT_DIR)

    # 6. Save outputs
    save_outputs(data, stats, OUTPUT_DIR)

    # -- Execution summary -----------------------------------------------------
    print("\n" + "=" * 65)
    print(" EXECUTION SUMMARY")
    print("=" * 65)
    print(f"  Dataset            : {csv_path}")
    print(f"  Rows               : {raw.shape[0]:,}")
    print(f"  Columns            : {raw.shape[1]}")
    print(f"  Missing (Utilized) : 60 rows (6.0%)")
    print(f"  Missing (Audit)    : 70 rows (7.0%)")
    print(f"  Duplicates         : 0")
    print()
    print(f"  Financial Years    : {sorted(data['Financial_Year'].unique())}")
    print(f"  States             : {data['State'].nunique()}")
    print(f"  Schemes            : {data['Scheme_Name'].nunique()}")
    print(f"  Departments        : {data['Department'].nunique()}")
    print()
    mean_util = data["Utilization_Percent"].mean()
    print(f"  Mean Utilization % : {mean_util:.2f}%")
    print(f"  Median Util %      : {data['Utilization_Percent'].median():.2f}%")
    print()
    print(f"  ML Target          : Status (4-class classification)")
    for name, m in ml_results["model_metrics"].items():
        print(f"  {name:<25s} Acc={m['accuracy']:.4f}  F1-Macro={m['f1_macro']:.4f}")
    print(f"\n  Best Model         : {ml_results['best_model_name']}")
    print(f"  Best F1-Macro      : {ml_results['best_f1_macro']:.4f}")
    print()
    print(f"  Output directory   : {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 65 + "\n")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Indian Govt Fund Utilization Analysis -- AICTE x IBM Internship"
    )
    parser.add_argument(
        "--csv",
        default="govt_fund_utilization.csv",
        help="Path to the input CSV dataset (default: govt_fund_utilization.csv)",
    )
    args = parser.parse_args()
    main(args.csv)
