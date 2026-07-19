"""
Fresham - Module 3: Spoilage-Risk Prediction
=================================================
Data Science / AI-ML deliverable for the TechyJaunt "Tomato Case Study" sprint.

What this script does
----------------------
1. Loads and cleans the harvest dataset (Harvest_Analysis_112045.xlsx).
2. Explores which pre-sale factors (storage, transport, harvest method, farm
   profile) drive tomato spoilage.
3. Trains and compares three regression models to predict spoilage_percentage(%).
4. Converts the regression output into a 3-tier risk label (Low / Medium / High)
   and trains a classifier that can be called in real time to produce a
   Sell / Store / Divert recommendation - the exact deliverable the PRD asks
   Module 3 to produce ("estimates days-until-spoilage and surfaces a
   sell/store/hold recommendation").
5. Runs the model against 3 realistic test scenarios (the PRD's minimum bar:
   "work correctly for at least 3 tested produce scenarios").
6. Saves every chart used in the presentation to ./charts/ and prints a
   metrics summary that the accompanying report quotes directly.

Run with: python3 spoilage_risk_model.py
Requires: pandas, numpy, scikit-learn, matplotlib, seaborn (all standard).
"""

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font="DejaVu Sans")
RANDOM_STATE = 42

DATA_PATH = "/mnt/user-data/uploads/Harvest_Analysis_112045.xlsx"
CHART_DIR = "charts"

# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_PATH)
df["spoilage_cause"] = df["spoilage_cause"].fillna("Unknown")

# ---------------------------------------------------------------------------
# 2. FEATURE / TARGET DEFINITION
# ---------------------------------------------------------------------------
# Predictors are restricted to information available BEFORE the produce is
# sold - i.e. decisions the app can influence. Columns like sellable_quantity_kg,
# revenue_naira, no_of_spoiled, days_to_sale and spoilage_cause are all
# *outcomes* of spoilage, not causes of it, so they are deliberately excluded
# to avoid target leakage.
NUMERIC_FEATURES = [
    "farm_size_hectares", "farmer_years_experience", "batch_quantity_kg",
    "storage_temperature_c", "storage_humidity_percent",
    "transport_duration_hours", "distance_km",
]
CATEGORICAL_FEATURES = [
    "harvest_method", "storage_type", "transportation_mode",
    "transport_condition", "market_destination", "Month Name",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_REG = "spoilage_percentage(%)"

X = df[FEATURES].copy()
y_reg = df[TARGET_REG].copy()

# 3-tier business risk label used for the Sell / Store / Divert recommendation
def risk_label(pct):
    if pct < 20:
        return "Low"
    elif pct < 40:
        return "Medium"
    else:
        return "High"

y_class = y_reg.apply(risk_label)

RECOMMENDATION = {
    "Low": "Sell as planned - spoilage risk is minimal, no special handling needed.",
    "Medium": "Move to cold storage and prioritize sale within the next 1-2 days.",
    "High": "Sell immediately at the nearest market or divert to processing/animal feed - "
            "holding this batch will likely destroy most of its value.",
}

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, yreg_train, yreg_test, yclass_train, yclass_test = train_test_split(
    X, y_reg, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

# ---------------------------------------------------------------------------
# 4. REGRESSION MODELS - predict spoilage_percentage(%)
# ---------------------------------------------------------------------------
reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=None, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE),
}

reg_results = {}
fitted_reg_pipelines = {}

for name, model in reg_models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, yreg_train)
    preds = pipe.predict(X_test)
    rmse = mean_squared_error(yreg_test, preds) ** 0.5
    mae = mean_absolute_error(yreg_test, preds)
    r2 = r2_score(yreg_test, preds)
    reg_results[name] = {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 4)}
    fitted_reg_pipelines[name] = pipe

best_reg_name = max(reg_results, key=lambda n: reg_results[n]["R2"])
best_reg_pipe = fitted_reg_pipelines[best_reg_name]
print("=== Regression model comparison (predicting spoilage_percentage) ===")
for name, m in reg_results.items():
    flag = "  <-- best" if name == best_reg_name else ""
    print(f"{name:20s} RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  R2={m['R2']:.4f}{flag}")

# ---------------------------------------------------------------------------
# 5. CLASSIFICATION MODEL - Sell / Store / Divert risk tier
# ---------------------------------------------------------------------------
clf_pipe = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")),
])
clf_pipe.fit(X_train, yclass_train)
class_preds = clf_pipe.predict(X_test)

clf_accuracy = accuracy_score(yclass_test, class_preds)
clf_f1_macro = f1_score(yclass_test, class_preds, average="macro")
clf_report = classification_report(yclass_test, class_preds, output_dict=True)
labels_order = ["Low", "Medium", "High"]
cm = confusion_matrix(yclass_test, class_preds, labels=labels_order)

print("\n=== Risk-tier classifier (Random Forest) ===")
print(f"Accuracy: {clf_accuracy:.4f}   Macro F1: {clf_f1_macro:.4f}")
print(classification_report(yclass_test, class_preds))

# ---------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (best regressor, if tree-based; else RF classifier)
# ---------------------------------------------------------------------------
def get_feature_names(ct):
    num_names = NUMERIC_FEATURES
    cat_names = list(ct.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + cat_names

importances_source = best_reg_pipe if hasattr(best_reg_pipe.named_steps["model"], "feature_importances_") else clf_pipe
feat_names = get_feature_names(importances_source.named_steps["prep"])
importances = importances_source.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(12)

# ---------------------------------------------------------------------------
# 7. THREE TEST SCENARIOS (PRD requirement: >= 3 tested produce scenarios)
# ---------------------------------------------------------------------------
scenarios = pd.DataFrame([
    {  # Scenario A: best-practice handling
        "farm_size_hectares": 4.0, "farmer_years_experience": 12, "batch_quantity_kg": 2000,
        "storage_temperature_c": 8, "storage_humidity_percent": 60,
        "transport_duration_hours": 2, "distance_km": 100,
        "harvest_method": "Hand", "storage_type": "Cold Storage",
        "transportation_mode": "Refrigerated", "transport_condition": "Good",
        "market_destination": "Kaduna", "Month Name": "March",
    },
    {  # Scenario B: typical / average handling
        "farm_size_hectares": 5.0, "farmer_years_experience": 15, "batch_quantity_kg": 2200,
        "storage_temperature_c": 24, "storage_humidity_percent": 72,
        "transport_duration_hours": 6, "distance_km": 450,
        "harvest_method": "Hand", "storage_type": "Evaporative Cooler",
        "transportation_mode": "Truck Covered", "transport_condition": "Moderate",
        "market_destination": "Lagos", "Month Name": "May",
    },
    {  # Scenario C: worst-case handling
        "farm_size_hectares": 6.0, "farmer_years_experience": 6, "batch_quantity_kg": 2800,
        "storage_temperature_c": 36, "storage_humidity_percent": 85,
        "transport_duration_hours": 12, "distance_km": 850,
        "harvest_method": "Mechanical", "storage_type": "Ambient",
        "transportation_mode": "Truck Open", "transport_condition": "Poor",
        "market_destination": "Port Harcourt", "Month Name": "June",
    },
])
scenario_labels = ["A - Best-practice handling", "B - Typical handling", "C - Worst-case handling"]
scenario_pred_pct = best_reg_pipe.predict(scenarios)
scenario_pred_class = clf_pipe.predict(scenarios)

print("\n=== Scenario tests (Module 3 acceptance criteria) ===")
for label, pct, cls in zip(scenario_labels, scenario_pred_pct, scenario_pred_class):
    print(f"{label:28s} -> predicted spoilage {pct:5.1f}%  | risk: {cls:6s} | {RECOMMENDATION[cls]}")

# ---------------------------------------------------------------------------
# 8. CHARTS FOR THE PRESENTATION
# ---------------------------------------------------------------------------
import os
os.makedirs(CHART_DIR, exist_ok=True)

# 8a. Correlation heatmap (numeric features + target)
plt.figure(figsize=(8, 6))
corr = df[NUMERIC_FEATURES + [TARGET_REG]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0)
plt.title("Correlation with Spoilage Percentage")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_correlation_heatmap.png", dpi=150)
plt.close()

# 8b. Spoilage by storage type
plt.figure(figsize=(7, 5))
order = df.groupby("storage_type")[TARGET_REG].median().sort_values().index
sns.boxplot(data=df, x="storage_type", y=TARGET_REG, order=order, palette="RdYlGn_r")
plt.title("Spoilage % by Storage Type")
plt.ylabel("Spoilage (%)")
plt.xlabel("")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_spoilage_by_storage.png", dpi=150)
plt.close()

# 8c. Spoilage by transport condition
plt.figure(figsize=(7, 5))
order = df.groupby("transport_condition")[TARGET_REG].median().sort_values().index
sns.boxplot(data=df, x="transport_condition", y=TARGET_REG, order=order, palette="RdYlGn_r")
plt.title("Spoilage % by Transport Condition")
plt.ylabel("Spoilage (%)")
plt.xlabel("")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_spoilage_by_transport.png", dpi=150)
plt.close()

# 8d. Model comparison bar chart
plt.figure(figsize=(7, 5))
names = list(reg_results.keys())
r2s = [reg_results[n]["R2"] for n in names]
bars = plt.bar(names, r2s, color=["#94a3b8", "#22c55e", "#3b82f6"])
for b, v in zip(bars, r2s):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
plt.ylabel("R\u00b2 (test set)")
plt.title("Regression Model Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_model_comparison.png", dpi=150)
plt.close()

# 8e. Actual vs predicted scatter (best model)
plt.figure(figsize=(6, 6))
preds_best = best_reg_pipe.predict(X_test)
plt.scatter(yreg_test, preds_best, alpha=0.3, s=15, color="#3b82f6")
lims = [0, max(yreg_test.max(), preds_best.max()) + 2]
plt.plot(lims, lims, "r--", linewidth=1)
plt.xlabel("Actual spoilage (%)")
plt.ylabel("Predicted spoilage (%)")
plt.title(f"Actual vs Predicted - {best_reg_name}")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_actual_vs_predicted.png", dpi=150)
plt.close()

# 8f. Feature importance
plt.figure(figsize=(8, 6))
plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#16a34a")
plt.title("Top Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/06_feature_importance.png", dpi=150)
plt.close()

# 8g. Confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel("Predicted risk tier")
plt.ylabel("Actual risk tier")
plt.title("Risk-Tier Classifier - Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/07_confusion_matrix.png", dpi=150)
plt.close()

# 8h. Risk tier distribution in dataset
plt.figure(figsize=(6, 5))
y_class.value_counts().reindex(labels_order).plot(kind="bar", color=["#22c55e", "#f59e0b", "#ef4444"])
plt.title("Risk Tier Distribution (full dataset)")
plt.ylabel("Number of batches")
plt.xlabel("")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_risk_distribution.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 9. SAVE A METRICS SUMMARY (consumed by the presentation/report)
# ---------------------------------------------------------------------------
summary = {
    "dataset_rows": len(df),
    "dataset_farmers": int(df["farmer_id"].nunique()),
    "regression_results": reg_results,
    "best_regression_model": best_reg_name,
    "classifier_accuracy": round(clf_accuracy, 4),
    "classifier_macro_f1": round(clf_f1_macro, 4),
    "classification_report": clf_report,
    "top_features": imp_df.to_dict(orient="records"),
    "scenarios": [
        {
            "label": label,
            "predicted_spoilage_pct": round(float(pct), 1),
            "risk_tier": cls,
            "recommendation": RECOMMENDATION[cls],
        }
        for label, pct, cls in zip(scenario_labels, scenario_pred_pct, scenario_pred_class)
    ],
}
with open("model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll charts saved to ./charts/  |  metrics saved to model_summary.json")
