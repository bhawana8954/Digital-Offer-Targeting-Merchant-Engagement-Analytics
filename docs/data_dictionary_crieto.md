# Data Dictionary — Criteo Uplift Dataset

For the Yelp Open Dataset dictionary, see `data_dictionary_yelp.md`.

**Source:** [criteo/criteo-uplift](https://huggingface.co/datasets/criteo/criteo-uplift) (Hugging Face), v2.1
**Scale:** 13,979,592 rows × 16 columns
**Data quality:** 0 missing values; 1,259,545 exact duplicate rows (expected — characteristic of high-frequency advertising log streams, not treated as an error)

| Column | Type | Description |
|---|---|---|
| `f0`–`f11` | float | 12 anonymized continuous features describing user/context characteristics |
| `treatment` | int (0/1) | Intent-to-treat flag — whether a bid was placed in the RTB auction |
| `exposure` | int (0/1) | Actual treatment received — whether the auction was won and the ad was successfully rendered (distinct from `treatment`) |
| `visit` | int (0/1) | Whether the user visited the advertiser's site |
| `conversion` | int (0/1) | Whether the user converted |

**Note on `treatment` vs `exposure`:** `treatment` reflects *intent-to-treat* (ITT) — assignment to the treatment arm regardless of outcome — while `exposure` reflects whether the ad was actually delivered. This distinction matters for uplift modeling, since ITT analysis avoids the selection bias that would come from conditioning on `exposure` (auction wins are not random).

## Derived: Development Sample

A 1% uniform random subset (139,796 rows, `random_state=42`) was drawn via `pandas.DataFrame.sample()` for fast iteration during EDA/debugging, before running final computations on the full dataset. Saved as `data/samples/criteo_sample.csv`.

## Derived: Incrementality Outputs

| File | Description |
|---|---|
| `outputs/tables/incrementality_summary.csv` | Dev-sample conversion & visit rates, control vs. treatment, with absolute/relative lift |
| `outputs/tables/segment_incrementality_summary.csv` | Full-dataset segment-level lift for the 4 selected features, restricted to segments meeting the "attractive segment" criteria (see `methodology_criteo.md`) |

**Segmentation features:** Of the 12 anonymized features (`f0`–`f11`), only **`f0`, `f2`, `f6`, `f8`** produced multiple usable quantile bins with meaningful conversion-rate variation under quartile-based binning. The remaining 8 features collapsed into a single effective bin (high value repetition) and were excluded from segment-level analysis. This is a screening result specific to this dataset's feature distributions, not a general claim about feature importance.

## Derived: Customer Targeting Model Outputs

| File | Description |
|---|---|
| `data/samples/customer_targeting_test_final.parquet` | Final v2 T-Learner test set (8,420 customers) with `p_treatment`, `p_control`, `uplift`, and `uplift_decile` columns — the scored, ranked output used for targeting evaluation |
| `outputs/tables/customer_targeting_model_evaluation.csv` | ROC-AUC and Log Loss for the treatment and control arm models individually (within-group conversion prediction quality — not an uplift-ranking metric) |