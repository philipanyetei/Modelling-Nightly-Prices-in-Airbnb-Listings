"""Step 5: Interactive market dashboard (Streamlit).

Run:  .venv/Scripts/python -m streamlit run src/05_dashboard.py

Views: price distribution, city/room-type comparison, availability,
review ratings, benchmark table (median & IQR by city x room_type),
and an interactive map. Filters: city, room type, area (geo cell),
price range, availability, review rating.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

st.set_page_config(page_title="Airbnb Price Dashboard", layout="wide")


@st.cache_data(show_spinner="Loading listings…")
def load_data():
    cols = [
        "id", "name", "city", "room_type", "property_type", "latitude",
        "longitude", "price", "accommodates", "bedrooms", "bathrooms",
        "beds", "availability_30", "availability_60", "availability_90",
        "availability_365", "has_availability", "number_of_reviews",
        "number_of_reviews_ltm", "reviews_per_month",
        "review_scores_rating", "host_is_superhost", "amenities_count",
        "has_wifi", "has_kitchen", "has_air_conditioning", "log_price",
        "price_outlier",
    ]
    df = pd.read_parquet(OUT / "cleaned" / "train_analysis.parquet",
                         columns=cols)
    df["geo_cell"] = (
        df["latitude"].round(2).astype(str) + ","
        + df["longitude"].round(2).astype(str)
    )
    return df


df = load_data()
st.title("Airbnb Nightly Price Dashboard")
st.caption(
    f"{len(df):,} listings · advertised nightly price · "
    "associations only — not realised revenue or causal effects"
)

# ---------------------------------------------------------- filters
f1, f2, f3 = st.columns(3)
city = f1.multiselect("City", sorted(df["city"].dropna().unique()))
room = f2.multiselect("Room type", sorted(df["room_type"].dropna().unique()))
price_rng = f3.slider("Price range ($)", 0, 1000, (0, 1000), 10)

f4, f5, f6 = st.columns(3)
avail_rng = f4.slider("Availability next 365d (days)", 0, 365, (0, 365), 5)
min_reviews = f5.slider("Minimum number of reviews", 0, 200, 0, 5)
rating_rng = f6.slider("Review rating", 0.0, 5.0, (0.0, 5.0), 0.1)

view = st.sidebar.radio(
    "View",
    ["Overview", "City & room-type comparison", "Availability",
     "Reviews & quality", "Benchmark table", "Map"],
)

# apply filters
mask = (
    df["price"].between(*price_rng)
    & df["availability_365"].between(*avail_rng)
    & (df["number_of_reviews"] >= min_reviews)
    & df["review_scores_rating"].between(*rating_rng, inclusive="both")
)
if city:
    mask &= df["city"].isin(city)
if room:
    mask &= df["room_type"].isin(room)
v = df[mask].copy()
st.sidebar.metric("Listings in view", f"{len(v):,}")


def seg_table(d, keys):
    t = (
        d.groupby(keys)
        .agg(
            listings=("price", "size"),
            median_price=("price", "median"),
            mean_price=("price", "mean"),
            q1=("price", lambda s: s.quantile(0.25)),
            q3=("price", lambda s: s.quantile(0.75)),
            median_availability_365=("availability_365", "median"),
            median_rating=("review_scores_rating", "median"),
            median_reviews=("number_of_reviews", "median"),
        )
        .reset_index()
    )
    t["iqr_price"] = t["q3"] - t["q1"]
    return t


# ------------------------------------------------------------- views
if view == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median price", f"${v['price'].median():.0f}")
    c2.metric("Mean price", f"${v['price'].mean():.0f}")
    c3.metric("IQR", f"${v['price'].quantile(0.75) - v['price'].quantile(0.25):.0f}")
    c4.metric("IQR outliers", f"{int(v['price_outlier'].sum()):,}")
    fig = px.histogram(v, x="price", nbins=80, color="room_type",
                       opacity=0.6, barmode="overlay",
                       title="Price distribution (filtered)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg_table(v, ["city", "room_type"]).head(50),
                 use_container_width=True)

elif view == "City & room-type comparison":
    cities = sorted(v["city"].unique())
    sel = st.multiselect("Cities to compare", cities,
                         default=cities[:8] if len(cities) > 8 else cities)
    vv = v[v["city"].isin(sel)] if sel else v
    fig = px.box(vv, x="room_type", y="price", color="city",
                 points=False, log_y=True,
                 category_orders={"room_type": [
                     "Shared room", "Private room", "Entire home/apt",
                     "Hotel room"]},
                 title="Price by room type and city (log scale)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg_table(vv, ["city", "room_type"]),
                 use_container_width=True)

elif view == "Availability":
    fig = px.box(v, x="city", y="availability_365", points=False,
                 title="Days available in next 365, by city")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.scatter(v, x="availability_365", y="price", color="room_type",
                      opacity=0.3,
                      title="Availability vs price (note: low availability "
                            "≠ proven demand — may be host blocks)")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(seg_table(v, ["city", "room_type"])[
        ["city", "room_type", "listings", "median_availability_365",
         "median_price"]], use_container_width=True)

elif view == "Reviews & quality":
    vv = v.dropna(subset=["review_scores_rating"])
    fig = px.scatter(vv.sample(min(20000, len(vv)), random_state=1),
                     x="review_scores_rating", y="price", color="room_type",
                     opacity=0.25, log_y=True,
                     title="Review rating vs price (sample)")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.scatter(vv.sample(min(20000, len(vv)), random_state=1),
                      x="number_of_reviews", y="price", color="room_type",
                      opacity=0.25, log_y=True,
                      title="Review count vs price (sample)")
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = px.bar(seg_table(vv, ["room_type"]), x="room_type",
                  y="median_rating", title="Median rating by room type")
    st.plotly_chart(fig3, use_container_width=True)

elif view == "Benchmark table":
    st.markdown("**Median and IQR by city × room type** — use this to judge "
                "whether a listing is priced unusually for its segment.")
    st.dataframe(seg_table(v, ["city", "room_type"]),
                 use_container_width=True)
    st.download_button("Download benchmark CSV",
                       seg_table(v, ["city", "room_type"]).to_csv(index=False),
                       "benchmark_city_roomtype.csv", "text/csv")

elif view == "Map":
    # aggregate filtered listings to ~1km cells
    cells = (
        v.groupby(["geo_cell", "latitude", "longitude", "city"],
                  observed=True)
        .agg(listings=("price", "size"), median_price=("price", "median"),
             median_rating=("review_scores_rating", "median"),
             median_availability=("availability_365", "median"))
        .reset_index()
    )
    cells = cells[cells["listings"] >= 5]
    fig = px.scatter_map(
        cells, lat="latitude", lon="longitude", color="median_price",
        size="listings", size_max=20, hover_name="city",
        hover_data={"geo_cell": True, "listings": True,
                    "median_price": True, "median_rating": True,
                    "median_availability": True},
        color_continuous_scale="Viridis", zoom=1.2, height=650,
        title="Median price by ~1 km area (≥5 listings in current filter)")
    fig.update_layout(map_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.caption(
    "Prices are advertised nightly rates. Availability may reflect host "
    "blocks, not demand. No revenue, fees, seasonality or costs are observed."
)
