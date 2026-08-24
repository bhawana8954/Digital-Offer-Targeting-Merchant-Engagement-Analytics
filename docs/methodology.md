> **Methodology / Implementation Note:** The Criteo uplift dataset source was changed from the Criteo AI Lab direct URL (which returned a 404) to **Hugging Face `criteo/criteo-uplift` v2.1**, downloaded using `hf_hub_download`. Additionally, the two-proportion z-test was implemented manually using **SciPy** instead of `statsmodels.stats.proportion.proportions_ztest` due to **Python 3.14 compatibility issues with Statsmodels**.


### Yelp Data Sampling

The Yelp Open Dataset totals several GB across its business, review, and
check-in files. To keep local development practical, filtering was
performed in a cloud notebook environment (Kaggle) rather than
downloading the full dataset. Businesses were filtered to the
Restaurants category within two metropolitan areas (Philadelphia and
Tampa), and reviews and check-ins were filtered to match. This yielded
8,812 businesses, 990,521 reviews, and check-in coverage for 8,583 of
those businesses — a sample large enough for meaningful trend and
engagement analysis while remaining tractable for local processing.