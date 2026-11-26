# Benchmarking and Modelling Nightly Prices in Airbnb Listings

**Unit of analysis:** individual listing · **Target:** advertised nightly price (`price`, USD)
**Data:** `rental price/airbnb_train.csv` — 208,856 listings across 108 cities (Inside Airbnb-style schema); `airbnb_test.csv` — 53,000 listings used only for out-of-sample prediction.

All scripts are in `src/` (`01_clean.py` → `05_dashboard.py`); all tables in `outputs/tables/`, figures in `outputs/figures/`.

---

## 1. Data quality and outlier report

See `outputs/reports/data_quality_report.md`. Key facts:

- Price was already numeric; **0 missing prices, 0 duplicate ids, 0 missing coordinates**.
- **17 implausible prices** (<$10 or >$2,000) removed from analysis.
- **14,242 statistical outliers (6.8%)** flagged via the IQR rule applied **within each city × room_type segment** — fences stored per listing (`seg_lower_fence`, `seg_upper_fence`). Outliers are flagged, not deleted: median outlier price is $496 vs $127 for non-outliers.
- **936 listings at exactly $1,000** — the observed price ceiling; treated as a platform cap.
- 40,646 listings (19.5%) have no review score — these are listings with no/few reviews; kept as "no score", never mean-imputed silently.
- Median and IQR are reported throughout because price is heavily right-skewed (overall median $136, mean $207).

## 2. How does median nightly price vary by city and area?

Full table: `outputs/tables/city_summary.csv`. Headline pattern (median, IQR):

| City | Listings | Median $ | IQR |
|---|---|---|---|
| Oslo | 526 | 835 | 691–950 |
| Copenhagen | 1,078 | 807 | 658–932 |
| Prague | 257 | 799 | 617–912 |
| Bangkok | 960 | 799 | 626–901 |
| Cape Town | 911 | 749 | 574–900 |
| Mexico City | 2,234 | 620 | 450–821 |

Premium coastal/luxury markets (Oslo, Copenhagen, Prague, Bangkok, Cape Town) sit far above the global median of $136. Because `neighborhood_overview` is free text rather than a label, neighbourhood-level comparison uses a derived ~1 km geo-cell (`outputs/tables/geo_cell_summary.csv`, cells with ≥10 listings). The interactive map (`outputs/dashboard/airbnb_price_map.html`, static fallback `outputs/figures/price_map_static.png`) shows price geography directly.

## 3. How do room types compare? (like-for-like)

Comparisons are only made **within the same city** (`outputs/tables/median_price_city_roomtype.csv`, heatmap `city_roomtype_heatmap.png`). Overall medians:

| Room type | Listings | Median $ | IQR |
|---|---|---|---|
| Entire home/apt | 163,059 | 151 | 96–265 |
| Hotel room | 1,211 | 220 | 116–402 |
| Private room | 43,773 | 79 | 52–145 |
| Shared room | 796 | 44 | 29–136 |

Entire homes command roughly double the median of private rooms; shared rooms are cheapest. Hotel rooms show the widest IQR — heterogeneous product. City × room-type medians vary enough that cross-type or cross-city raw comparisons would be misleading.

## 4. Which features are associated with higher prices?

Correlations with log-price (`outputs/tables/feature_correlations.csv`) and the regression both point the same way:

- **Size/capacity dominate**: `accommodates` (r≈0.41), `bedrooms` (0.37), `bathrooms` (0.32), `beds` (0.31). In the model: each additional guest ≈ +9.1%, bedroom ≈ +8.1%, bathroom ≈ +9.6% (ceteris paribus, log points).
- **Amenities**: `amenities_count` r≈0.15; each extra amenity ≈ +0.35%. Wi-Fi/kitchen/A/C flags included in the model.
- **Location**: city fixed effects are the largest single block of explanatory power (e.g., Prague ≈ +1,141% vs the baseline city).
- **Reviews**: overall rating correlates weakly (+0.08), but the like-for-like check is clearer — listings rated 4.8–5.0 sit **+4.9% above** their city×room-type median log-price, while ratings <4.0 sit ~5–9% below (`rating_band_summary.csv`). `review_scores_location` is the strongest sub-score (r≈0.14).
- **Review count**: essentially flat to slightly negative (r≈−0.07); listings with 50+ reviews are ~5.6% *below* segment median — likely older, established, competitively priced listings.
- **Availability**: `estimated_occupancy_l365d` correlates −0.12 with price — higher-priced listings show *lower* estimated occupancy in this data (association only; the occupancy field is modelled, not observed).

## 5. Are there extreme or implausible prices?

Yes — 6.8% of listings fall outside their segment's IQR fences; the extreme tail is real but bounded by the $1,000 cap. 331 listings under $20 are mostly shared/private rooms in low-cost markets — plausible, retained. Full detail in `data_quality_report.md` and `top_outliers.csv`.

## 6. How does availability vary across locations?

`outputs/tables/availability_by_city.csv`: median days available in the next 365 range from **87 (Copenhagen)** to **358 (Istanbul)**. Copenhagen, Amsterdam, Edinburgh, Dublin and Prague — all high-price markets — cluster at the *low*-availability end, consistent with tighter listed supply in premium markets. By room type, shared rooms are most available (median 325d) and entire homes least (229d). **Caveat:** availability mixes bookings, host blocks and minimum-stay rules — it cannot be read as demand.

## 7. Are review ratings and counts associated with price?

Weakly positive for *quality*, ~zero for *quantity* — see §4. Scatter plots in `reviews_vs_price.png`. The cleanest evidence is the within-segment rating-band analysis: a 4.8+ rating is worth roughly +5% versus the same city's room-type median.

## 8. Can a baseline model estimate a reasonable price?

OLS on **ln(price)**, 80/20 holdout (`outputs/reports/model_report.md`, `model_metrics.csv`):

| Model | MAE ($) | RMSE ($) | log-MAE | log-RMSE |
|---|---|---|---|---|
| OLS log-price | 77.76 | 444.87 | 0.370 | 0.487 |
| OLS capped at $1,000 | **74.13** | **128.70** | 0.370 | 0.487 |
| Median by city×room_type | 88.40 | 151.43 | 0.453 | 0.595 |
| Global median | 124.66 | 209.76 | — | — |

The log-linear model beats the required benchmark on **every** metric. Without capping, exponentiating log predictions produces implausible values up to ~$27k (RMSE 445); capping at the observed $1,000 ceiling fixes this and improves all metrics — a documented, defensible post-processing step. R² = 0.64. Test-set predictions for all 52,805 usable test listings are in `outputs/predictions/airbnb_test_predictions.csv`.

## 9. Pricing recommendations (for a property manager)

1. **Benchmark within segment first.** Compare a listing's price to the median/IQR for its *own city × room_type* (dashboard → Benchmark table). Deviations beyond Q3+1.5·IQR warrant review — but may be justified by unobserved quality.
2. **Capacity and space are the strongest levers** consistent with higher prices: each additional guest/bedroom/bathroom corresponds to roughly +8–10%, holding location constant.
3. **Top-rated listings (≥4.8) price ~5% above segment median** — review quality is associated with a modest premium; investing in cleanliness/accuracy/location scores is plausibly worthwhile, but this is an association, not a causal effect.
4. **Do not read low availability as high demand** — it may reflect host blocks; verify against actual booking data before adjusting price.
5. **Use the model as a sanity check**: a listing priced far above its model prediction should be justified by amenities, views, or quality not captured in the data.

## 10. Limitations — what this dataset cannot measure

- **No realised outcomes**: price is the *advertised* rate, not revenue, occupancy or bookings. `estimated_occupancy_l365d`/`estimated_revenue_l365d` are review-based *estimates*, not observations.
- **No costs**: cleaning fees, service fees, taxes, utilities, furnishing, licensing — profitability cannot be assessed.
- **No seasonality/calendar**: a single snapshot; no dates, events, or demand shocks.
- **No property quality**: photos, renovation, design, views are absent — the largest likely driver of residual price variance.
- **Neighbourhood is weak**: `neighborhood_overview` is free text; geo-cells are a coarse proxy.
- **Causal claims are impossible**: coefficients describe associations. Changing a feature (e.g., adding a bathroom) does not guarantee the associated price change.
- **Coverage bias**: 108 cities of wildly different sizes; tail cities have too few listings for reliable segment benchmarks.
- **Platform cap**: prices appear capped at $1,000, truncating the true upper tail.
