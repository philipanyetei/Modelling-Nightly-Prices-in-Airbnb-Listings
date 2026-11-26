"""Step 3: Geographic analysis.

Interactive Plotly map of median nightly price by ~1km geo-cell
(cells with >=10 listings), plus a static matplotlib fallback PNG.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(
    OUT / "cleaned" / "train_analysis.parquet",
    columns=["id", "name", "city", "room_type", "latitude", "longitude",
             "price", "availability_365", "review_scores_rating",
             "number_of_reviews", "accommodates"],
)
df["geo_cell"] = (
    df["latitude"].round(2).astype(str) + "," + df["longitude"].round(2).astype(str)
)

cells = (
    df.groupby("geo_cell")
    .agg(
        city=("city", "first"),
        listings=("id", "count"),
        median_price=("price", "median"),
        mean_price=("price", "mean"),
        median_availability_365=("availability_365", "median"),
        median_rating=("review_scores_rating", "median"),
        median_reviews=("number_of_reviews", "median"),
        median_accommodates=("accommodates", "median"),
        lat=("latitude", "median"),
        lon=("longitude", "median"),
    )
    .reset_index()
)
cells = cells[cells["listings"] >= 10].sort_values("median_price")
print("map cells:", len(cells))

# ---------------- interactive map ----------------
fig = px.scatter_map(
    cells,
    lat="lat",
    lon="lon",
    color="median_price",
    size="listings",
    size_max=25,
    hover_name="city",
    hover_data={
        "geo_cell": True,
        "listings": True,
        "median_price": True,
        "mean_price": True,
        "median_availability_365": True,
        "median_rating": True,
        "median_reviews": True,
        "lat": False,
        "lon": False,
    },
    color_continuous_scale="Viridis",
    zoom=1.2,
    height=750,
    title="Median nightly Airbnb price by ~1 km area "
          "(areas with ≥10 listings; bubble size = listing count)",
)
fig.update_layout(
    map_style="open-street-map",
    coloraxis_colorbar=dict(title="median $/night"),
)
fig.write_html(OUT / "dashboard" / "airbnb_price_map.html")

# ---------------- static fallback ----------------
fig2, ax = plt.subplots(figsize=(12, 7))
sc = ax.scatter(
    cells["lon"], cells["lat"],
    c=cells["median_price"], s=np.sqrt(cells["listings"]) * 8,
    cmap="viridis", alpha=0.75, edgecolors="none",
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Median nightly price by ~1 km area (≥10 listings); "
             "darker/larger = higher price / more listings")
ax.set_aspect("equal")
fig2.colorbar(sc, ax=ax, label="median $/night")
plt.tight_layout()
plt.savefig(FIG / "price_map_static.png", dpi=200)
plt.close()

# city-level choropleth-style summary for report
city_geo = (
    df.groupby("city")
    .agg(listings=("id", "count"), median_price=("price", "median"),
         lat=("latitude", "median"), lon=("longitude", "median"))
    .reset_index()
)
city_geo.to_csv(OUT / "tables" / "city_coordinates.csv", index=False)
print("done")
