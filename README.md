# TomatoLink — Spoilage-Risk Prediction (Module 3)

Machine learning module for **TomatoLink**, a platform reducing post-harvest tomato losses in Nigeria. Given a batch's storage and transport conditions, the model predicts spoilage percentage and returns a **Sell / Store / Divert** recommendation a farmer can act on immediately.

Built for the TechyJaunt Sprint — Tomato Case Study.

## Overview

| | |
|---|---|
| **Problem** | ~30–40% of harvested tomatoes in Nigeria spoil before reaching market |
| **Approach** | Regression model predicts spoilage %, mapped to a 3-tier risk classifier |
| **Dataset** | 10,000 harvest batches from 1,000 farmers (Jan–Jun 2026) |
| **Best model** | Gradient Boosting Regressor — R² 0.644, MAE 6.8 pts |
| **Classifier** | Random Forest, 3-class (Low/Medium/High) — 64.9% accuracy, macro F1 0.655 |

## Repository contents

```
spoilage_risk_model.py      # full pipeline: EDA → training → evaluation → scenarios → charts
model_summary.json          # generated metrics (created on run)
charts/                     # generated figures (created on run)
```

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

Place `Harvest_Analysis_112045.xlsx` alongside the script, or update `DATA_PATH` at the top of `spoilage_risk_model.py`.

## Usage

```bash
python3 spoilage_risk_model.py
```

This will:
1. Load and clean the dataset
2. Train/compare 3 regression models for `spoilage_percentage(%)`
3. Train a Random Forest risk-tier classifier (Low / Medium / High)
4. Run 3 test scenarios (best-practice, typical, worst-case handling)
5. Save 8 charts to `./charts/`
6. Save all metrics to `model_summary.json`

## Features used

Only information available **before** a batch is sold — no target leakage:

- `farm_size_hectares`, `farmer_years_experience`, `batch_quantity_kg`
- `harvest_method` (Hand / Mechanical)
- `storage_type`, `storage_temperature_c`, `storage_humidity_percent`
- `transportation_mode`, `transport_duration_hours`, `transport_condition`
- `distance_km`, `market_destination`, `Month Name`

Excluded on purpose: `sellable_quantity_kg`, `revenue_naira`, `no_of_spoiled`, `days_to_sale`, `spoilage_cause` — these are all *outcomes* of spoilage, not causes, and aren't known at prediction time.

## Results

### Regression — predicting spoilage %

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 8.586 | 6.983 | 0.6233 |
| Random Forest | 8.617 | 7.002 | 0.6205 |
| **Gradient Boosting** | **8.352** | **6.798** | **0.6435** |

### Risk classifier — Sell / Store / Divert

| Tier | Spoilage range | Recommendation | Precision | Recall |
|---|---|---|---|---|
| Low | < 20% | Sell as planned | 0.70 | 0.65 |
| Medium | 20–40% | Store in cold storage, sell within 1–2 days | 0.63 | 0.62 |
| High | > 40% | Sell immediately or divert to processing | 0.64 | 0.70 |

Accuracy: **64.9%** · Macro F1: **0.655**. Errors mostly fall between adjacent tiers (Low↔Medium, Medium↔High) rather than opposite extremes — the safer failure mode for this use case.

### Top drivers of spoilage

1. **Storage temperature** — strongest single predictor
2. **Storage type (Ambient vs. Cold Storage/Evaporative Cooler)** — near-equal weight
3. Transport duration and condition
4. Storage humidity

Storage temperature and storage type alone account for ~77% of the model's predictive weight — the clearest signal for where the product should push farmer behavior (cold-chain adoption).

### Scenario tests

| Scenario | Predicted spoilage | Risk | Recommendation |
|---|---|---|---|
| A — Best-practice handling | 12.1% | Low | Sell as planned |
| B — Typical handling | 24.1% | Medium | Move to cold storage, sell within 1–2 days |
| C — Worst-case handling | 51.0% | High | Sell immediately or divert to processing |

Satisfies the PRD requirement of correctly handling ≥3 tested produce scenarios.

## Limitations

- R² of 0.64 means ~36% of spoilage variance comes from factors not in this dataset (variety, ripeness at harvest, in-transit weather, handling damage) — expected for an MVP.
- Classifier accuracy should improve as real usage data comes in post-launch (see Phase 3 of the roadmap: spoilage-risk model refinement).
- Dataset spans January–June only; performance in peak/rainy-season months is unvalidated.

## Next steps

- Wire the trained classifier into the live prototype (Day 3 AI Integration checkpoint)
- Log predictions vs. actual outcomes post-launch to build a retraining dataset
- Surface a confidence indicator in the UI (e.g. "based on 10,000 similar batches")

## AI Usage Log note

This script was built with AI assistance for pipeline structure and chart generation. Every metric reported here was verified by running the script against the real dataset — none were estimated. The feature list was manually reviewed to confirm no outcome/leakage columns were included as predictors.
