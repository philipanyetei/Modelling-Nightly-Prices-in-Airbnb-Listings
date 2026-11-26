"""Step 1: Load, inspect and clean the Airbnb listings data.

Outputs:
  outputs/tables/missingness.csv        - missing-value report (train)
  outputs/tables/data_quality_summary.csv - key data-quality counts
  outputs/cleaned/train.parquet         - cleaned training listings
  outputs/cleaned/test.parquet          - cleaned test listings (no price)
  outputs/reports/data_quality_report.md - human-readable report
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "rental price"
OUT = ROOT / "outputs"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "cleaned").mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(parents=True, exist_ok=True)

DATE_COLS = ["host_since", "first_review", "last_review"]
PCT_COLS = ["host_response_rate", "host_acceptance_rate"]

AMENITY_FLAGS = {
    "has_wifi": r"wifi|wireless internet",
    "has_kitchen": r"kitchen",
    "has_air_conditioning": r"air conditioning|air conditioning",
    "has_parking": r"parking|car park",
    "has_washer": r"washer|washing machine|laundry",
    "has_pool": r"pool",
    "has_hot_tub": r"hot tub|jacuzzi",
    "has_gym": r"gym|fitness",
    "has_ev_charger": r"ev charger|electric vehicle",
    "has_beach_access": r"beach",
    "has_dishwasher": r"dishwasher",
    "has_tv": r'"tv"|cable tv|netflix',
}


def to_numeric(series):
    return pd.to_numeric(
        series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )


def parse_amenities(series):
    """Amenities are stored like {"A","B"} or {} ; return count + flag columns."""
    s = series.fillna("").astype(str).str.strip()
    # split on commas inside braces; entries are quoted with ""
    entries = s.str.findall(r'"([^"]*)"')
    counts = entries.apply(len)
    joined = entries.apply(lambda xs: "|".join(xs).lower())
    flags = {}
    for name, pat in AMENITY_FLAGS.items():
        flags[name] = joined.str.contains(pat, regex=True, na=False).astype("int8")
    return counts.rename("amenities_count"), pd.DataFrame(flags, index=series.index)


def clean_frame(df, is_train):
    df = df.copy()

    # --- identifiers / text tidiness ---
    df["city"] = df["city"].astype("string").str.strip()
    df["room_type"] = df["room_type"].astype("string").str.strip()
    df["property_type"] = df["property_type"].astype("string").str.strip()
    df.loc[df["property_type"].isin(["", "nan", "None"]), "property_type"] = pd.NA
    df["neighbourhood"] = (
        df.get("neighborhood_overview")
        .astype("string")
        .str.split(r"(?<=[.!?])\s+")
        .str[0]
        .str.slice(0, 120)
    )

    # --- numeric coercions ---
    df["price"] = to_numeric(df["price"]) if is_train else np.nan
    for col in ["accommodates", "bedrooms", "beds", "availability_30",
                "availability_60", "availability_90", "availability_365",
                "availability_eoy", "number_of_reviews", "number_of_reviews_ltm",
                "number_of_reviews_l30d", "number_of_reviews_ly",
                "reviews_per_month", "host_listings_count",
                "host_total_listings_count", "latitude", "longitude",
                "estimated_occupancy_l365d", "estimated_revenue_l365d"] + \
               [f"review_scores_{s}" for s in
                ["rating", "accuracy", "cleanliness", "checkin",
                 "communication", "location", "value"]]:
        if col in df.columns:
            df[col] = to_numeric(df[col])
    df["bathrooms"] = to_numeric(df["bathrooms"])

    for col in PCT_COLS:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    df["has_availability"] = (
        df["has_availability"].astype("string").str.lower().isin(["t", "true", "1"])
    )

    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- engineered variables ---
    if is_train:
        df["log_price"] = np.log(df["price"].where(df["price"] > 0))
    df["host_experience_years"] = (
        (pd.Timestamp("2025-01-01") - df["host_since"]).dt.days / 365.25
    ).clip(lower=0)

    amen_count, amen_flags = parse_amenities(df["amenities"])
    df["amenities_count"] = amen_count
    for c in amen_flags.columns:
        df[c] = amen_flags[c]

    df["price_per_person"] = df["price"] / df["accommodates"].where(df["accommodates"] > 0)
    df["reviews_per_month"] = to_numeric(df["reviews_per_month"])

    # --- plausibility flags ---
    df["implausible_price"] = (
        df["price"].isna() | (df["price"] <= 0) | (df["price"] < 10) | (df["price"] > 2000)
    ) if is_train else False
    df["invalid_coords"] = (
        df["latitude"].isna() | df["longitude"].isna()
        | (df["latitude"].abs() > 90) | (df["longitude"].abs() > 180)
    )
    df["review_score_missing"] = df["review_scores_rating"].isna()
    df["no_reviews"] = to_numeric(df["number_of_reviews"]).fillna(0) == 0

    return df


def iqr_outlier_flags(df):
    """Flag statistical outliers within each city x room_type segment."""
    g = df.groupby(["city", "room_type"])["price"]
    q1 = g.transform("quantile", 0.25)
    q3 = g.transform("quantile", 0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["seg_q1"] = q1
    df["seg_q3"] = q3
    df["seg_iqr"] = iqr
    df["seg_lower_fence"] = lo
    df["seg_upper_fence"] = hi
    df["price_outlier"] = (
        ((df["price"] < lo) | (df["price"] > hi)) & df["price"].notna()
    ).astype("int8")
    return df


def main():
    train = pd.read_csv(DATA / "airbnb_train.csv", low_memory=False)
    test = pd.read_csv(DATA / "airbnb_test.csv", low_memory=False)
    print("raw shapes:", train.shape, test.shape)

    # --- missingness report (raw) ---
    miss = train.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / len(train) * 100).round(2)
    missing_df = pd.DataFrame({"missing": miss, "pct": miss_pct})
    missing_df[missing_df["missing"] > 0].to_csv(OUT / "tables" / "missingness.csv")

    train = clean_frame(train, is_train=True)
    test = clean_frame(test, is_train=False)
    train = iqr_outlier_flags(train)

    # --- analysis-ready subset: drop rows unusable for price analysis ---
    analysis = train[
        train["price"].notna()
        & (train["price"] > 0)
        & ~train["implausible_price"]
        & train["city"].notna()
        & train["room_type"].notna()
        & ~train["invalid_coords"]
    ].copy()
    print("analysis rows:", len(analysis), "of", len(train))

    # --- summary stats for report ---
    summ = {
        "raw_train_rows": len(train),
        "analysis_rows": len(analysis),
        "dropped_price": int(train["price"].isna().sum() + (train["price"] <= 0).sum()),
        "implausible_price_flagged": int(train["implausible_price"].sum()),
        "invalid_coords": int(train["invalid_coords"].sum()),
        "missing_price": int(train["price"].isna().sum()),
        "missing_city": int(train["city"].isna().sum()),
        "missing_room_type": int(train["room_type"].isna().sum()),
        "missing_rating": int(train["review_scores_rating"].isna().sum()),
        "missing_bedrooms": int(train["bedrooms"].isna().sum()),
        "missing_bathrooms": int(train["bathrooms"].isna().sum()),
        "missing_beds": int(train["beds"].isna().sum()),
        "duplicate_ids": int(train["id"].duplicated().sum()),
        "price_iqr_outliers": int(train["price_outlier"].sum()),
        "price_at_1000": int((train["price"] == 1000).sum()),
        "price_below_20": int((train["price"] < 20).sum()),
        "cities": int(train["city"].nunique()),
        "room_types": ", ".join(sorted(train["room_type"].dropna().unique())),
    }
    pd.DataFrame([summ]).T.to_csv(OUT / "tables" / "data_quality_summary.csv")

    analysis.to_parquet(OUT / "cleaned" / "train_analysis.parquet", index=False)
    train.to_parquet(OUT / "cleaned" / "train_full.parquet", index=False)
    test.to_parquet(OUT / "cleaned" / "test.parquet", index=False)
    print(summ)


if __name__ == "__main__":
    main()
