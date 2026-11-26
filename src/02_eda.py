"""Step 2: Exploratory analysis answering the business questions.

Q1 median price by city / neighbourhood-ish area
Q2 room-type comparison (like-for-like within city)
Q3 features associated with price
Q4 extreme/implausible prices (uses flags from cleaning)
Q5 availability across locations
Q6 review ratings/counts vs price

Outputs: outputs/tables/*.csv and outputs/figures/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
df = pd.read_parquet(OUT / "cleaned" / "train_analysis.parquet")

# coarse geographic cell (~1 km) as a neighbourhood proxy
df["geo_cell"] = (
    df["latitude"].round(2).astype(str) + "," + df["longitude"].round(2).astype(str)
)

TOP_CITIES = df["city"].value_counts().head(15).index.tolist()


def q(x, p):
    return x.quantile(p)


# ---------------------------------------------------------------- Q1: cities
def city_summary(d):
    return d.groupby("city").agg(
        listings=("id", "count"),
        median_price=("price", "median"),
        mean_price=("price", "mean"),
        q1=("price", lambda s: s.quantile(0.25)),
        q3=("price", lambda s: s.quantile(0.75)),
        median_availability_365=("availability_365", "median"),
        median_rating=("review_scores_rating", "median"),
        median_reviews=("number_of_reviews", "median"),
    ).reset_index()


city_sum = city_summary(df)
city_sum["iqr_price"] = city_sum["q3"] - city_sum["q1"]
city_sum = city_sum.sort_values("median_price", ascending=False)
city_sum.to_csv(TAB / "city_summary.csv", index=False)

# geo-cell summary for the biggest markets (neighbourhood proxy)
geo = (
    df[df["city"].isin(TOP_CITIES)]
    .groupby(["city", "geo_cell"])
    .agg(
        listings=("id", "count"),
        median_price=("price", "median"),
        median_availability_365=("availability_365", "median"),
    )
    .reset_index()
)
geo = geo[geo["listings"] >= 10].sort_values("median_price", ascending=False)
geo.to_csv(TAB / "geo_cell_summary.csv", index=False)

# ------------------------------------------------------- Q2: room types
def seg_summary(d, keys):
    return (
        d.groupby(keys)
        .agg(
            listings=("id", "count"),
            median_price=("price", "median"),
            mean_price=("price", "mean"),
            q1=("price", lambda s: s.quantile(0.25)),
            q3=("price", lambda s: s.quantile(0.75)),
            median_availability_365=("availability_365", "median"),
            median_rating=("review_scores_rating", "median"),
        )
        .reset_index()
    )


seg = seg_summary(df, ["city", "room_type"])
seg["iqr_price"] = seg["q3"] - seg["q1"]
seg.to_csv(TAB / "segment_summary.csv", index=False)  # benchmark table

room_sum = seg_summary(df, ["room_type"])
room_sum["iqr_price"] = room_sum["q3"] - room_sum["q1"]
room_sum.to_csv(TAB / "room_type_summary.csv", index=False)

# median price pivot: city x room_type (top 15 cities)
pivot = (
    df[df["city"].isin(TOP_CITIES)]
    .pivot_table(index="city", columns="room_type", values="price",
                 aggfunc="median")
    .round(0)
)
pivot.to_csv(TAB / "median_price_city_roomtype.csv")

# ------------------------------------------------- Q3: feature associations
num_feats = [
    "accommodates", "bedrooms", "bathrooms", "beds", "amenities_count",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "availability_eoy", "number_of_reviews", "number_of_reviews_ltm",
    "number_of_reviews_l30d", "number_of_reviews_ly", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "host_listings_count", "host_total_listings_count",
    "host_experience_years", "estimated_occupancy_l365d",
    "estimated_revenue_l365d",
    # NB: price_per_person is derived from the target and excluded to avoid
    # a tautological correlation.
]
feat_rows = []
for c in num_feats:
    s = df[c]
    feat_rows.append({
        "feature": c,
        "n_nonnull": int(s.notna().sum()),
        "corr_log_price": float(df["log_price"].corr(s)),
        "median": float(s.median()),
    })
feat_df = pd.DataFrame(feat_rows).sort_values("corr_log_price",
                                              key=abs, ascending=False)
feat_df.to_csv(TAB / "feature_correlations.csv", index=False)

# amenity on/off comparison (raw medians; like-for-like caveat applies)
amen_cols = [c for c in df.columns if c.startswith("has_") and c != "has_availability"]
amen_rows = []
for c in amen_cols:
    for val, lab in [(1, "with"), (0, "without")]:
        sub = df[df[c] == val]
        amen_rows.append({
            "amenity": c, "group": lab, "n": len(sub),
            "median_price": sub["price"].median(),
            "median_log_price": sub["log_price"].median(),
        })
pd.DataFrame(amen_rows).to_csv(TAB / "amenity_comparison.csv", index=False)

# ------------------------------------------------------------- Q4: outliers
outl = df[df["price_outlier"] == 1]
outlier_summary = pd.DataFrame([{
    "total_listings": len(df),
    "iqr_outliers": len(outl),
    "pct_outliers": round(100 * len(outl) / len(df), 2),
    "outlier_median_price": outl["price"].median(),
    "nonoutlier_median_price": df.loc[df["price_outlier"] == 0, "price"].median(),
    "price_at_cap_1000": int((df["price"] == 1000).sum()),
    "price_below_20": int((df["price"] < 20).sum()),
}])
outlier_summary.to_csv(TAB / "outlier_summary.csv", index=False)
df[df["price_outlier"] == 1].nlargest(200, "price")[
    ["id", "name", "city", "room_type", "property_type", "price",
     "accommodates", "review_scores_rating", "number_of_reviews",
     "seg_upper_fence"]
].to_csv(TAB / "top_outliers.csv", index=False)

# ------------------------------------------------------------- Q5: availability
avail_cols = ["availability_30", "availability_60", "availability_90", "availability_365"]
avail_city = (
    df.groupby("city")[avail_cols + ["price"]]
    .median()
    .reset_index()
    .sort_values("availability_365")
)
avail_city.to_csv(TAB / "availability_by_city.csv", index=False)

avail_room = (
    df.groupby("room_type")[avail_cols + ["price"]]
    .agg({**{c: "median" for c in avail_cols}, "price": "median"})
    .reset_index()
)
avail_room.to_csv(TAB / "availability_by_roomtype.csv", index=False)

# ------------------------------------------------------------- Q6: reviews
rev = df.dropna(subset=["review_scores_rating"])
rev_summary = rev.groupby("room_type").agg(
    listings=("id", "count"),
    median_rating=("review_scores_rating", "median"),
    median_reviews=("number_of_reviews", "median"),
    median_reviews_per_month=("reviews_per_month", "median"),
    median_price=("price", "median"),
).reset_index()
rev_summary.to_csv(TAB / "review_summary.csv", index=False)

# rating band vs price (like-for-like: within city×room_type, use demeaned log price)
work = df.dropna(subset=["review_scores_rating"]).copy()
work["log_price_dm"] = work["log_price"] - work.groupby(
    ["city", "room_type"])["log_price"].transform("median")
work["rating_band"] = pd.cut(
    work["review_scores_rating"],
    bins=[0, 3.5, 4.0, 4.5, 4.8, 5.01],
    labels=["<3.5", "3.5-4.0", "4.0-4.5", "4.5-4.8", "4.8-5.0"],
)
work["reviews_band"] = pd.cut(
    work["number_of_reviews"],
    bins=[-1, 0, 5, 20, 50, np.inf],
    labels=["0", "1-5", "6-20", "21-50", "50+"],
)
work.groupby("rating_band", observed=True).agg(
    n=("id", "count"),
    median_price=("price", "median"),
    median_log_price_dm=("log_price_dm", "median"),
).to_csv(TAB / "rating_band_summary.csv")
work.groupby("reviews_band", observed=True).agg(
    n=("id", "count"),
    median_price=("price", "median"),
    median_log_price_dm=("log_price_dm", "median"),
).to_csv(TAB / "reviews_band_summary.csv")

# ============================================================== figures
# F1: price distribution overall + by room type
d = df[df["price"] <= 1000]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(np.log10(d["price"]), bins=60, color="#4C72B0", edgecolor="white")
axes[0].set_xlabel("log10 nightly price ($)")
axes[0].set_ylabel("Listings")
axes[0].set_title("Price distribution is strongly right-skewed")
sns.boxplot(data=d, x="room_type", y="price", ax=axes[1],
            order=["Shared room", "Private room", "Entire home/apt", "Hotel room"],
            showfliers=False)
axes[1].set_yscale("log")
axes[1].set_title("Price by room type (log scale, outliers hidden)")
axes[1].set_xlabel("")
plt.tight_layout()
plt.savefig(FIG / "price_distribution.png", dpi=200)
plt.close()

# F2: city median prices with IQR (top 15 by listings)
top = city_sum[city_sum["city"].isin(TOP_CITIES)].sort_values("median_price")
fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(top))
ax.barh(y, top["median_price"], color="#55A868", label="median")
ax.errorbar(top["median_price"], y,
            xerr=[top["median_price"] - top["q1"], top["q3"] - top["median_price"]],
            fmt="none", ecolor="black", capsize=3, label="Q1-Q3")
ax.set_yticks(y, top["city"])
ax.set_xlabel("Nightly price ($)")
ax.set_title("Median nightly price by city (top 15 markets, whiskers = IQR)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "city_median_prices.png", dpi=200)
plt.close()

# F3: heatmap median price city x room_type
hm = pivot.astype(float).fillna(0)
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(hm, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
            cbar_kws={"label": "median price ($)"})
ax.set_title("Median nightly price: city × room type")
ax.set_xlabel("Room type")
ax.set_ylabel("City")
plt.tight_layout()
plt.savefig(FIG / "city_roomtype_heatmap.png", dpi=200)
plt.close()

# F4: availability
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df[df["city"].isin(TOP_CITIES)], x="city", y="availability_365",
            ax=axes[0], showfliers=False, order=TOP_CITIES)
axes[0].tick_params(axis="x", rotation=45)
axes[0].set_title("Days available in next 365 (top 15 cities)")
sns.boxplot(data=df, x="room_type", y="availability_365", ax=axes[1],
            showfliers=False,
            order=["Shared room", "Private room", "Entire home/apt", "Hotel room"])
axes[1].set_title("Availability by room type")
plt.tight_layout()
plt.savefig(FIG / "availability.png", dpi=200)
plt.close()

# F5: reviews vs price
sample = df.dropna(subset=["review_scores_rating"]).sample(
    min(20000, len(df)), random_state=42)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(data=sample, x="review_scores_rating", y="log_price",
                hue="room_type", alpha=0.25, s=8, ax=axes[0], legend=False)
axes[0].set_title("Review rating vs log price")
axes[0].set_xlabel("Review rating")
axes[0].set_ylabel("ln(price)")
sns.scatterplot(data=sample, x="number_of_reviews", y="log_price",
                hue="room_type", alpha=0.25, s=8, ax=axes[1],
                legend=False)
axes[1].set_xscale("log")
axes[1].set_title("Review count vs log price")
axes[1].set_xlabel("Number of reviews (log)")
axes[1].set_ylabel("ln(price)")
plt.tight_layout()
plt.savefig(FIG / "reviews_vs_price.png", dpi=200)
plt.close()

# F6: amenity premiums (median log price with vs without, top amenities)
amen_piv = (
    pd.DataFrame(amen_rows)
    .pivot_table(index="amenity", columns="group",
                 values="median_log_price")
)
common = amen_piv[(amen_piv["with"].notna()) & (amen_piv["without"].notna())]
common["lift"] = common["with"] - common["without"]
common = common.sort_values("lift", ascending=False).head(8)
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(common))
ax.bar(x - 0.2, common["without"], 0.4, label="without", color="#B0B7C3")
ax.bar(x + 0.2, common["with"], 0.4, label="with", color="#4C72B0")
ax.set_xticks(x, common.index.str.replace("has_", "", regex=False), rotation=30)
ax.set_ylabel("Median ln(price)")
ax.set_title("Median log price by amenity (raw comparison — not causal)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "amenity_premiums.png", dpi=200)
plt.close()

# F7: outlier visualization — price vs segment fence for one big market
sub = df[df["city"] == "London"]
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(np.log10(sub["price"]), bins=80, color="#8172B3", edgecolor="white")
ax.axvline(np.log10(sub.loc[sub["room_type"] == "Entire home/apt",
                            "seg_upper_fence"].median()),
           color="red", ls="--", label="median upper fence (Entire home/apt)")
ax.set_xlabel("log10 nightly price ($)")
ax.set_ylabel("Listings")
ax.set_title("London price distribution with IQR outlier fence")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "outlier_fences.png", dpi=200)
plt.close()

print("tables:", sorted(p.name for p in TAB.glob("*.csv")))
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
