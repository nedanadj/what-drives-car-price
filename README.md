# What Drives the Price of a Car?

**Practical Application II — CRISP-DM applied to a used-car pricing problem**

Full analysis: [`prompt_II.ipynb`](./prompt_II.ipynb)

## Business Understanding

A used-car dealership wants to know what consumers value in a used car so it can fine-tune its inventory. This is framed as a supervised regression problem: predict `price` from vehicle attributes (age, mileage, manufacturer, condition, fuel type, drivetrain, transmission, body type, title status, paint color, and state), then interpret the fitted model's coefficients to rank which attributes move price up or down.

## Data

Source: a cleaned subset (426,880 rows) of a larger Kaggle used-vehicle listings dataset, provided as `data/vehicles.csv`.

Key data quality issues identified and addressed:
- `size` (72% missing), `cylinders` (42%), `condition` (41%), `VIN` (38%), `drive`/`paint_color` (~31%), `type` (22%) had substantial missingness.
- `id` and `VIN` are identifiers with no predictive value; `model` (29,649 unique values) and `region` (404 unique values) were too high-cardinality to one-hot encode directly.
- `price` contained values of $0 and up to $3.7 billion; `year` ranged back to 1900; `odometer` reached 10,000,000 miles — all clear data-entry errors.

## Data Preparation

- Dropped `id`, `VIN`, `model`, `region`, `size`.
- Filtered to `price` in [$500, $150,000], `year` in [1990, 2022], `odometer` in [0, 300,000] miles — retaining 86% of rows.
- Engineered `age = 2022 - year`.
- Imputed missing `odometer` with the median; recoded missing categorical values as an explicit `"missing"` level rather than dropping rows.
- Built an `sklearn` `ColumnTransformer` (standard-scaled numeric features + one-hot encoded categoricals) inside a `Pipeline` to prevent train/test leakage.

## Modeling

Three regression models were fit and tuned with `GridSearchCV` and cross-validation:

| Model | Test RMSE | Test R² | Test MAE |
|---|---|---|---|
| Linear Regression | ~$8,616 | 0.656 | ~$5,658 |
| **Ridge (alpha=1)** — final model | ~$8,616 | 0.656 | ~$5,658 |
| Lasso (alpha=1) | ~$8,620 | 0.656 | ~$5,658 |

Ridge was selected as the final model: it edges out the others slightly on test RMSE, keeps every feature's coefficient visible (unlike Lasso, which zeroes some out), and its coefficients are directly interpretable in dollar terms.

**Why RMSE:** it's in dollars (the unit the client thinks in) and penalizes large mispricings more than small ones, which matters most for margin. MAE is reported alongside as a more robust "typical error" figure.

## Key Findings

**Raises price, all else equal:** lower mileage, newer model year, diesel fuel, higher cylinder counts (8+), 4-wheel drive, "new" condition, certain premium/EV/truck-oriented brands, manual transmission (small, enthusiast-driven effect).

**Lowers price:** higher mileage, older model year, salvage/parts-only title, poor/salvage condition, several mass-market economy brands, electric/hybrid fuel (in this largely pre-2022 dataset), bus/hatchback body styles.

The model explains about 66% of price variation; the remaining ~34% reflects factors not in this dataset (trim level, accident history, local supply/demand, negotiation).

## Recommendations

1. Build a simple internal pricing calculator from the final Ridge model.
2. Capture `condition`, `cylinders`, and `drive` consistently at intake — these were missing for 30–40% of training listings and are likely the highest-leverage data fix available.
3. Refresh the model periodically on recent data, since used-EV pricing in particular has moved a lot since this dataset was collected.
4. Consider separate models per vehicle segment (trucks/SUVs vs. sedans vs. luxury) in a future iteration.
5. Prioritize newer, lower-mileage, clean-title, 4WD inventory with stronger engines for the safest margins; be more cautious stocking high-mileage economy sedans, salvage-title vehicles, or older used EVs unless priced accordingly.

## Repo structure

```
├── README.md
├── prompt_II.ipynb       # full analysis notebook (CRISP-DM), pre-run with outputs
├── data/
│   └── vehicles.csv
└── images/
    ├── kurt.jpeg
    └── crisp.png
```

> Note: `data/vehicles.csv` is ~51MB. If your GitHub workflow or grader prefers a lighter repo, it's fine to add `data/` to `.gitignore` and instead link to the original Kaggle dataset in this README — the notebook only needs the file present locally to re-run.
