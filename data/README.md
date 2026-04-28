# Data Directory

## sample_data.csv

A synthetic customer churn dataset used solely to validate the ML pipeline.

**Features:**
| Column | Type | Description |
|---|---|---|
| customer_id | string | Unique identifier (dropped before training) |
| age | numeric | Customer age in years |
| tenure_months | numeric | Months as customer |
| monthly_spend | numeric | Average monthly spend (USD) |
| num_products | numeric | Number of products held |
| num_support_tickets | numeric | Support tickets raised in last 6 months |
| region | categorical | Geographic region |
| account_type | categorical | Account tier |
| payment_method | categorical | Payment method used |
| churned | binary (0/1) | **Target** — 1 = churned, 0 = retained |

**Important:** This dataset is entirely synthetic. Metrics reported on this data do not reflect real-world performance. Replace with real data for production use.

## Replacing the Dataset

1. Place your CSV file in this directory.
2. Update `configs/config.yaml`:
   - `dataset.dataset_path`
   - `dataset.target_column`
   - `features.numeric_features` (or leave empty for auto-detection)
   - `features.categorical_features` (or leave empty for auto-detection)
3. Re-run `make train`.
