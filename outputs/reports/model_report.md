# Baseline regression report

Formula: `log_price ~ C(room_type) + C(property_type)
    + accommodates + bedrooms + bathrooms + beds
    + availability_365 + review_scores_rating + has_reviews
    + np.log1p(number_of_reviews)
    + C(host_is_superhost) + C(host_response_time)
    + C(city) + amenities_count
    + has_wifi + has_kitchen + has_air_conditioning`

Training rows: 166,464; validation rows: 41,617

## Holdout metrics (price scale)

| model                               |      MAE |    RMSE |     logMAE |    logRMSE |
|:------------------------------------|---------:|--------:|-----------:|-----------:|
| OLS log-price                       |  77.7623 | 444.87  |   0.369894 |   0.486565 |
| OLS log-price (capped at train max) |  74.1321 | 128.695 |   0.369894 |   0.486565 |
| median by city x room_type          |  88.4038 | 151.429 |   0.453321 |   0.594585 |
| global median                       | 124.665  | 209.759 | nan        | nan        |

## Strongest positive associations (log-price coefficients)

|                       |    coef |   std_err |       t |            p |   exp_coef_pct |
|:----------------------|--------:|----------:|--------:|-------------:|---------------:|
| Intercept             | 3.56215 | 0.0879892 | 40.484  | 0            |       3423.9   |
| C(city)[T.Taipei]     | 2.52633 | 0.0779079 | 32.4272 | 5.96836e-230 |       1150.76  |
| C(city)[T.Prague]     | 2.51882 | 0.0682634 | 36.8986 | 7.81984e-297 |       1141.4   |
| C(city)[T.Oslo]       | 2.33663 | 0.0637535 | 36.651  | 6.58324e-293 |        934.629 |
| C(city)[T.Cape Town]  | 2.27025 | 0.0617947 | 36.7385 | 2.71375e-294 |        868.179 |
| C(city)[T.Istanbul]   | 2.248   | 0.062794  | 35.7996 | 1.31315e-279 |        846.877 |
| C(city)[T.Copenhagen] | 2.23115 | 0.0615385 | 36.2562 | 1.05546e-286 |        831.058 |
| C(city)[T.Bangkok]    | 2.21552 | 0.061605  | 35.9633 | 3.84226e-282 |        816.619 |

## Strongest negative associations

|                                               |      coef |   std_err |         t |           p |   exp_coef_pct |
|:----------------------------------------------|----------:|----------:|----------:|------------:|---------------:|
| C(property_type)[T.Campsite]                  | -0.934032 | 0.103128  |  -9.05704 | 1.35433e-19 |       -60.7034 |
| C(room_type)[T.Shared room]                   | -0.911371 | 0.0789993 | -11.5364  | 8.87866e-31 |       -59.8027 |
| C(property_type)[T.Room in hostel]            | -0.493372 | 0.107223  |  -4.60138 | 4.20019e-06 |       -38.9436 |
| C(city)[T.Riga]                               | -0.449868 | 0.063684  |  -7.06408 | 1.62318e-12 |       -36.2288 |
| C(property_type)[T.Private room in home]      | -0.415827 | 0.0716192 |  -5.80609 | 6.40654e-09 |       -34.0206 |
| C(city)[T.Athens]                             | -0.4029   | 0.0599288 |  -6.72299 | 1.78605e-11 |       -33.1621 |
| C(property_type)[T.Private room in bungalow]  | -0.358963 | 0.080922  |  -4.43591 | 9.1741e-06  |       -30.16   |
| C(property_type)[T.Private room in townhouse] | -0.338831 | 0.073057  |  -4.6379  | 3.52235e-06 |       -28.7397 |

Coefficients are on ln(price): a coefficient of b implies an approximate 100·(e^b − 1)% difference in price, ceteris paribus. These are associations, not causal effects.
