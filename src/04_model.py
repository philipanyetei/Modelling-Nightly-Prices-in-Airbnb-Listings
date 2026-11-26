"""Step 4: Baseline regression on log(price).

- OLS via statsmodels formula API
- 80/20 train/validation split (random_state=42)
- Metrics on price scale: MAE, RMSE
- Benchmark: median price by city x room_type (fit on train split only)
- Final model refit on all training data -> predictions for airbnb_test.csv

Outputs:
  outputs/tables/model_metrics.csv
  outputs/tables/model_coefficients.csv
  outputs/tables/validation_predictions.csv (sample)
  outputs/predictions/airbnb_test_predictions.csv
  outputs/reports/model_report.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "predictions").mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(parents=True, exist_ok=True)

FEATURES = [
    "price", "room_type", "property_type", "accommodates", "bedrooms",
    "bathrooms", "beds", "availability_365", "review_scores_rating",
    "number_of_reviews", "host_is_superhost", "host_response_time",
    "city", "amenities_count", "has_wifi", "has_kitchen",
    "has_air_conditioning",
]

FORMULA = """
    log_price ~ C(room_type) + C(property_type)
    + accommodates + bedrooms + bathrooms + beds
    + availability_365 + review_scores_rating + has_reviews
    + np.log1p(number_of_reviews)
    + C(host_is_superhost) + C(host_response_time)
    + C(city) + amenities_count
    + has_wifi + has_kitchen + has_air_conditioning
"""


def prepare(df):
    d = df[FEATURES + ["log_price", "id"]].copy()
    # review rating missing == listing has no reviews -> segment-median impute
    d["has_reviews"] = d["review_scores_rating"].notna().astype(int)
    seg_med = (
        d.dropna(subset=["review_scores_rating"])
        .groupby(["city", "room_type"])["review_scores_rating"]
        .median()
    )
    overall_med = d["review_scores_rating"].median()
    seg_dict = {(str(c), str(r)): v for (c, r), v in seg_med.items()}
    fill = pd.Series([
        seg_dict.get((str(c), str(r)), overall_med)
        for c, r in zip(d["city"], d["room_type"])
    ], index=d.index)
    d["review_scores_rating"] = d["review_scores_rating"].fillna(fill)
    d["host_is_superhost"] = d["host_is_superhost"].astype("string").fillna("f")
    d["host_response_time"] = d["host_response_time"].astype("string").fillna("unknown")
    d["room_type"] = d["room_type"].astype("string")
    d["property_type"] = d["property_type"].astype("string").fillna("unknown")
    d["city"] = d["city"].astype("string")

    # collapse rare categorical levels so train/validation/test share levels
    for col, min_count in [("property_type", 30), ("city", 30),
                           ("room_type", 30), ("host_response_time", 5),
                           ("host_is_superhost", 5)]:
        counts = d[col].value_counts()
        rare = set(counts[counts < min_count].index)
        if rare:
            d[col] = d[col].where(~d[col].isin(rare), "other")
    return d


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
    }


def main():
    df = pd.read_parquet(OUT / "cleaned" / "train_analysis.parquet")
    data = prepare(df)
    # rows needed for the model must be complete on regressors
    data = data.dropna(subset=[
        "price", "accommodates", "bedrooms", "bathrooms", "beds",
        "availability_365", "amenities_count",
    ])
    print("model rows:", len(data))

    train, val = train_test_split(data, test_size=0.20, random_state=42)

    model = smf.ols(FORMULA, data=train).fit()
    print(f"R-squared: {model.rsquared:.4f}  Adj.R2: {model.rsquared_adj:.4f}  "
          f"n params: {len(model.params)}  df_model: {model.df_model}")

    val = val.copy()
    val["pred_log_price"] = model.predict(val)
    val["pred_price"] = np.exp(val["pred_log_price"])
    price_cap = float(train["price"].max())
    val["pred_price_clip"] = val["pred_price"].clip(upper=price_cap)

    reg = metrics(val["price"], val["pred_price"])
    reg_clip = metrics(val["price"], val["pred_price_clip"])
    reg_log = {
        "logMAE": mean_absolute_error(val["log_price"], val["pred_log_price"]),
        "logRMSE": float(root_mean_squared_error(val["log_price"], val["pred_log_price"])),
    }

    # benchmark: median price by city x room_type, fit on TRAIN split only
    bench = train.groupby(["city", "room_type"])["price"].median().rename("bench_price")
    val = val.join(bench, on=["city", "room_type"])
    val["bench_price"] = val["bench_price"].fillna(train["price"].median())
    bm = metrics(val["price"], val["bench_price"])
    bm_log = {
        "logMAE": mean_absolute_error(np.log(val["price"]), np.log(val["bench_price"])),
        "logRMSE": float(root_mean_squared_error(np.log(val["price"]), np.log(val["bench_price"]))),
    }

    naive = metrics(val["price"], pd.Series(train["price"].median(), index=val.index))

    res = pd.DataFrame([
        {"model": "OLS log-price", **reg, **reg_log},
        {"model": "OLS log-price (capped at train max)", **reg_clip, **reg_log},
        {"model": "median by city x room_type", **bm, **bm_log},
        {"model": "global median", **naive,
         "logMAE": np.nan, "logRMSE": np.nan},
    ])
    res.to_csv(OUT / "tables" / "model_metrics.csv", index=False)
    print(res)

    # coefficients (top by |t|)
    coef = model.params.rename("coef").to_frame()
    coef["std_err"] = model.bse
    coef["t"] = model.tvalues
    coef["p"] = model.pvalues
    coef["exp_coef_pct"] = (np.exp(coef["coef"]) - 1) * 100
    coef.sort_values("t", key=abs, ascending=False).to_csv(
        OUT / "tables" / "model_coefficients.csv")

    val[["id", "city", "room_type", "price", "pred_price", "bench_price",
         "pred_log_price"]].head(500).to_csv(
        OUT / "tables" / "validation_predictions_sample.csv", index=False)

    # ---- final model on all data -> test predictions ----
    final_model = smf.ols(FORMULA, data=data).fit()
    test = pd.read_parquet(OUT / "cleaned" / "test.parquet")
    tdata = test.copy()
    tdata["log_price"] = np.nan  # placeholder column not needed for predict
    tprep = tdata.reindex(columns=data.columns.drop("price").tolist() + ["price"])
    # rebuild same engineered fields used in prepare()
    tprep = test.copy()
    tprep["has_reviews"] = tprep["review_scores_rating"].notna().astype(int)
    seg_med_t = (
        tprep.dropna(subset=["review_scores_rating"])
        .groupby(["city", "room_type"])["review_scores_rating"].median()
    )
    seg_dict_t = {(str(c), str(r)): v for (c, r), v in seg_med_t.items()}
    fill_t = pd.Series([
        seg_dict_t.get((str(c), str(r)), tprep["review_scores_rating"].median())
        for c, r in zip(tprep["city"], tprep["room_type"])
    ], index=tprep.index)
    tprep["review_scores_rating"] = tprep["review_scores_rating"].fillna(fill_t)
    for c in ["host_is_superhost", "host_response_time", "room_type",
              "property_type", "city"]:
        tprep[c] = tprep[c].astype("string")
    tprep["host_is_superhost"] = tprep["host_is_superhost"].fillna("f")
    tprep["host_response_time"] = tprep["host_response_time"].fillna("unknown")
    tprep["property_type"] = tprep["property_type"].fillna("unknown")
    # map any level unseen in training to "other" (patsy requires known levels)
    for col in ["room_type", "property_type", "city", "host_is_superhost",
                "host_response_time"]:
        known = set(data[col].dropna().unique())
        tprep[col] = tprep[col].astype("string").where(
            tprep[col].isin(known), "other")
    tprep = tprep.dropna(subset=[
        "accommodates", "bedrooms", "bathrooms", "beds",
        "availability_365", "amenities_count"])
    try:
        tprep["pred_log_price"] = final_model.predict(tprep)
    except Exception as e:
        # unseen categorical levels: predict row-wise is slow; instead align via patsy
        print("batch predict failed:", e)
        tprep["pred_log_price"] = [
            final_model.predict(tprep.iloc[[i]]).iloc[0]
            for i in range(len(tprep))
        ]
    preds = pd.DataFrame({
        "id": tprep["id"].astype(int),
        # cap at observed train maximum ($1000) — the data's price ceiling
        "price": np.round(np.exp(tprep["pred_log_price"]).clip(
            upper=float(train["price"].max())), 2),
    })
    preds.to_csv(OUT / "predictions" / "airbnb_test_predictions.csv", index=False)
    print("test predictions:", preds.shape, "price range:",
          preds["price"].min(), "-", preds["price"].max())

    # ---- model report ----
    top_pos = coef.sort_values("coef", ascending=False).head(8)
    top_neg = coef.sort_values("coef").head(8)
    with open(OUT / "reports" / "model_report.md", "w") as f:
        f.write("# Baseline regression report\n\n")
        f.write(f"Formula: `{FORMULA.strip()}`\n\n")
        f.write(f"Training rows: {len(train):,}; validation rows: {len(val):,}\n\n")
        f.write("## Holdout metrics (price scale)\n\n")
        f.write(res.to_markdown(index=False) + "\n\n")
        f.write("## Strongest positive associations (log-price coefficients)\n\n")
        f.write(top_pos.to_markdown() + "\n\n")
        f.write("## Strongest negative associations\n\n")
        f.write(top_neg.to_markdown() + "\n\n")
        f.write(
            "Coefficients are on ln(price): a coefficient of b implies an "
            "approximate " + "100·(e^b − 1)% difference in price, ceteris paribus. "
            "These are associations, not causal effects.\n"
        )
    print("done")


if __name__ == "__main__":
    main()
