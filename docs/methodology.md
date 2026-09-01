# Methodology

## Criteo Uplift Workstream

### 1. Data Integrity Audit
Before any analysis, the full Criteo dataset was audited for:
- **Missing values** — confirmed zero `NaN` values across all 16 columns.
- **Duplicates** — 1,259,545 exact duplicate rows identified. These were not dropped, as they are consistent with the nature of high-frequency advertising log data (repeated identical impressions/events are expected, not data errors).
- **Binary integrity** — `treatment`, `visit`, and `conversion` were programmatically verified to contain only `{0, 1}` values.

### 2. Group Balance & Distribution Checks
Treatment/control group sizes were assessed via raw counts and percentage shares. Conditional conversion distributions were computed independently within each treatment bucket using `pd.crosstab(..., normalize="index").mul(100)`, to allow clean comparison of conversion rates without conflating group size with outcome rate.

### 3. Development Sampling Strategy
A 1% development subset (139,796 rows) was drawn with a fixed seed (`random_state=42`) to:
- Speed up iteration during exploratory analysis and pipeline debugging
- Guarantee reproducibility across environments/team members
- Defer full-dataset (13.98M row) computation to final runs only

### 4. Global Incrementality Testing

**Dev-sample check:** Conversion and visit lift (absolute + relative) were first computed on the 1% dev sample as a quick directional check (control vs. treatment groups), before committing to full-dataset computation.

**Full-dataset statistical validation:** Using the complete 13.98M-row dataset, conversion counts were aggregated by treatment group (loading only the `treatment`/`conversion` columns as `int8` to keep memory low), then tested with a manually implemented **two-proportion z-test**:

- **H₀:** p_treatment = p_control (no difference in conversion rates)
- **H₁:** p_treatment ≠ p_control (two-sided, α = 0.05)
- Pooled proportion and pooled standard error computed from combined treatment+control conversions
- Z-statistic = (rate difference) / (pooled SE); two-sided p-value via `scipy.stats.norm.sf`
- 95% CI computed separately using the **unpooled** standard error (standard practice: pooled SE for the hypothesis test, unpooled SE for the CI)
- Implemented manually with `numpy` + `scipy.stats.norm` (no external stats library) since `statsmodels` was unavailable — see setup notes in README

**Result:** Control conversion rate 0.194% vs. treatment 0.309% — an absolute lift of 0.115 percentage points (59.4% relative lift), Z = 28.52, p ≈ 7.3e-179, 95% CI [0.108%, 0.122%]. Statistically significant, though the effect size is small in absolute terms — commercial viability depends on weighing this against campaign cost and ROAS, since a 13.98M-row sample can make even small differences statistically significant without necessarily being commercially meaningful.

### 5. Heterogeneous Treatment Effect (Segment-Level) Analysis

**Goal:** Identify customer segments (via features `f0`–`f11`) with above-average incremental response, since the global lift is a population average that can mask meaningful variation.

**Process:**
1. **Feature exploration** — examined distributions, missingness, and cardinality of `f0`–`f11`.
2. **Binning** — attempted quartile-based binning (`pd.qcut`) on all 12 features; checked the number of *effective* (non-collapsed) bins per feature. Most features collapsed to a single bin due to repeated values; only `f0`, `f2`, `f6`, `f8` retained multiple usable bins.
3. **Segment-level lift** — for each bin of each of the 4 selected features, computed segment population, treatment/control conversion counts and rates, absolute/relative lift, plus a per-segment two-proportion z-test (same method as above) with 95% CI.
4. **"Attractive segment" filter** — a segment was flagged only if it met **all three** criteria:
   - Absolute lift above the mean lift across all evaluated segments
   - Adequate sample size (≥100 in *both* treatment and control within the segment)
   - Statistically reliable (p < 0.05 **and** 95% CI excludes zero)
5. **Output** — 4 segments passed all three filters (one each from `f8`, `f6`, `f0`, `f2`), ranging from +0.142 to +0.404 percentage points absolute lift.

**Caveats (carried into the findings):**
- The RCT design gives causal proof only at the population level; the selected features are descriptive markers of *where* response is highest, not independently proven causal drivers.
- Testing ~10 segments at α = 0.05 carries multiple-testing risk (elevated false-positive probability); these segments are exploratory and should be validated on a holdout sample or a future campaign before being operationalized.

### 6. Customer-Level Uplift Modeling — T-Learner (v1)

**Goal:** Move from segment-level heterogeneity to individual customer-level uplift scoring, using a **T-Learner**: two independent logistic regression models (treatment-arm and control-arm), scored on the same customers, with uplift = P(convert | T=1) − P(convert | T=0).

**Setup:**
- Used the 1% dev sample (139,796 rows); all 12 `f0`–`f11` features as predictors, `conversion` as label.
- **Downsampled treatment to match control** (21,048 each) to avoid the model learning treatment-group imbalance instead of true response patterns, then shuffled into a balanced dataset.
- **80/20 train/test split, stratified on treatment×conversion strata** (not just treatment) to preserve the already-rare conversion cases in both splits.
- Trained `model_treatment` and `model_control` separately (`LogisticRegression(max_iter=1000, class_weight="balanced")`), each only on its own arm's training rows.
- Scored both models on the full test set; uplift = `p_treatment − p_control` per customer.
- Ranked customers into uplift deciles (`pd.qcut`, 10 bins) and compared **predicted** uplift (model score) against **observed** uplift (actual treatment-vs-control conversion rate within each decile) — the standard uplift-model evaluation approach (a Qini-style decile check).

**Findings (v1):**
- Predicted uplift ranking was clean and monotonic by construction, but **observed uplift was not monotonic** across deciles — e.g. decile 8 showed positive observed uplift despite a negative predicted score.
- **Root cause — extreme label sparsity:** only 437 conversions across 139,796 dev-sample rows; the held-out test set had just 20 conversions total (13 treatment / 7 control) across 8,420 rows, leaving many deciles with 0–3 conversions — enough for a single customer to swing an entire decile's observed rate.
- **Root cause — model miscalibration:** predicted uplift magnitudes (range −0.76 to +0.98) were 2–3 orders of magnitude larger than the true population-level lift (~0.00115 from notebook 2's full-dataset test). With only 52 treatment-arm and 27 control-arm positive cases spread across 12 continuous features, `class_weight="balanced"` + zero regularization caused quasi-complete separation — the model fit overconfidently to a handful of rare positives rather than learning genuine treatment-effect heterogeneity.

**Conclusion:** This v1 T-Learner is diagnostic only — the ranking reflects overfitting, not real uplift heterogeneity, so it isn't a trustworthy targeting model as-is. It correctly identifies *why* it fails (sparsity + no regularization + balanced-class distortion), which motivates v2's fixes: feature standardization, proper regularization, and removing artificial class balancing. See `04_customer_targeting_model_v2.ipynb`.

### 7. Customer-Level Uplift Modeling — T-Learner v2 (Corrected)

**Goal:** Fix the calibration failures identified in v1 (overconfident, poorly scaled uplift predictions) and establish whether the model beats a random-ranking baseline.

**Changes from v1:**
- **Random baseline established first** — before retraining, a random score was assigned to each test customer and ranked into deciles, to get a non-predictive reference point. Observed uplift across random deciles fluctuated with no consistent pattern (as expected), confirming that any structure the real model shows should be compared against this noise floor, not against zero.
- **Removed `class_weight="balanced"`** — this alone reduced predicted uplift from a ±0.76/+0.98 range down to a ~-0.50/+0.22 range, but tail predictions still far exceeded the true population effect (~0.115 pp).
- **Regularization sweep** — compared `C = 1.0, 0.1, 0.01` (inverse regularization strength; lower = stronger penalty). Std. deviation of predicted uplift dropped from 0.0108 (C=1.0) to 0.0033 (C=0.01), with C=0.01 giving the most stable distribution. **C = 0.01 selected** for the final model, though regularization alone didn't fully fix calibration.
- **Feature standardization** — `StandardScaler` fit on training features, applied consistently to treatment/control/test splits, combined with the chosen regularization for the final model.

**Final v2 result:**
- Mean predicted uplift (~0.111 pp) now closely matches the true population-level effect (~0.115 pp from notebook 2's full-dataset test) — a major calibration improvement over v1.
- Still has extreme tail predictions (-0.169 to +0.186), so outputs are best used as a **relative ranking score**, not an absolute per-customer treatment-effect estimate.
- **Top decile (Rank 1)** achieved observed uplift of +0.567 pp vs. the random baseline's +0.289 pp for its own top decile — and captured **65% of all 20 test-set conversions** (13 of 20), a meaningful concentration signal.
- **Full decile ranking is still non-monotonic** — e.g. rank 10 (lowest predicted uplift) had the highest observed uplift, most likely driven by the same conversion sparsity (20 total conversions) rather than a real inversion of effect.

**Verdict:** Exploratory but promising — the model shows real signal at the top decile (well above random), but the extreme sparsity of conversions (20 in the test set) makes anything beyond rank-1 targeting unreliable. Not yet suitable for operational deployment without a larger validation sample, cross-validation, and more robust uplift methods (e.g. X-Learner, causal forests). Final scored test set saved to `data/samples/customer_targeting_test_final.parquet`.

### 8. Model Evaluation & Targeting Recommendation

**Within-group model quality (not uplift quality):** The treatment and control arm models were each evaluated independently on their own conversion-prediction task using ROC-AUC (ranking ability) and Log Loss (probability calibration quality) — Treatment: AUC 0.9468, Log Loss 0.0141; Control: AUC 0.9400, Log Loss 0.0088. A confusion matrix was deliberately not used, since conversion is rare enough that a default 0.5 threshold would classify almost everyone as a non-converter and provide little insight.

**Important distinction:** Strong ROC-AUC/Log Loss on the individual treatment/control models does **not** imply strong uplift-ranking performance, since uplift is the *difference* of two predicted probabilities — each model can rank within-group conversion likelihood well while their difference still fails to reliably separate high- vs. low-uplift customers. This is consistent with the non-monotonic decile pattern seen in notebook 4, and is treated as a caution rather than a contradiction.

**Percentile-based targeting groups:** Rather than relying on noisy individual deciles (few conversions each), customers were grouped into wider percentile bands by predicted uplift — Top 10%, 10–25%, 25–50%, Bottom 50% — to get larger, more stable sample sizes per group. Predicted uplift decreased monotonically across these bands (0.0072 → 0.0002), but **observed uplift did not** (Top 10%: +0.0057; 10–25%: −0.0001; 25–50%: −0.0000; Bottom 50%: +0.0015) — the Bottom 50% figure is flagged as unreliable since it's driven by zero observed control-group conversions in that band, not genuine incremental response.

**Cumulative targeting scenarios:** Evaluated observed lift for "top N%" cutoffs (10/20/30%) against a full-population random-targeting baseline (0.0014):
| Target Group | Observed Lift | vs. Random Baseline |
|---|---|---|
| Top 10% | 0.0057 | ~4× higher |
| Top 20% | 0.0028 | ~2× higher |
| Top 30% | 0.0012 | Below baseline |

**Business recommendation:** Prioritize the Top 10% of customers by predicted uplift for targeted offers; Top 20% is a reasonable expansion if broader reach is needed; targeting beyond Top 30% shows no evidence of benefit over random targeting. The model is best framed as a **precision targeting tool for the highest-uplift segment**, not a reliable ranking across the full population — consistent with all prior notebooks' caveats about conversion sparsity limiting confidence outside the top tier.

## Yelp Merchant Engagement Workstream:

### 1.Yelp Sampling

**Execution environment:** run as a Kaggle Notebook against the Yelp Open Dataset mounted server-side — avoids downloading the ~5.3GB review.json locally.

**City/category selection:** businesses filtered where categories contains "Restaurants" (case-insensitive substring match, not exact-category equality) → 52,268 restaurant businesses. Ranked by restaurant count per city; Philadelphia (5,852) and Tampa (2,960) selected as the top two cities.

**Business sample:** 8,812 businesses, 0 duplicate business_ids.

**Review sample:** review.json streamed in 100k-row chunks (memory constraint — file too large to load in one pass) and matched against the business_id set incrementally. Result: 990,521 reviews, all 8,812 sampled businesses represented.

**Checkin sample:** filtered directly (small file, no chunking needed). Result: 8,583 businesses with check-in history — 229 of the 8,812 sampled businesses have no check-in record at all (expected gap, not a data quality issue).

**Referential integrity checks:** post-filter, verified 0 orphan business_ids in both the review sample and the checkin sample (i.e., every ID in those samples traces back to the business sample).

**Kaggle → local transfer:** the review sample (400MB+ as a single parquet) wouldn't reliably download as one file from the Kaggle UI, so it was split into 10 row-chunked parquet parts (~100k rows / ~40MB each) for transfer, to be recombined locally via glob + concat. Business and checkin samples were small enough to download directly as CSV.

### 2.Merchant Engagement Table

**Inputs combined:** business sample (8,812 rows), review sample (10 parquet parts reloaded via glob + concat, 990,521 rows), check-in sample (8,583 rows). All joined on business_id, left-joined onto the business table so every one of the 8,812 businesses is retained.

**Recent vs. earlier window:** defined relative to the max review date in the dataset (2022-01-19), not the run date — recent = last 6 months (2021-07-19 to 2022-01-19), earlier = the preceding 6 months (2021-01-19 to 2021-07-19). Reviews/check-ins outside both windows are labeled "outside" and excluded from trend calculations (used only for lifetime totals).

**Check-in event expansion:** raw check-in rows store a single comma-separated string of all timestamps per business. Before aggregation, this string is split and exploded into one row per check-in event, then parsed to datetime (unparseable timestamps dropped) — 1,796,822 individual check-in events recovered from 8,583 businesses.

**Growth vs. change:** two parallel metrics computed for reviews and check-ins:
- `*_growth` — percent change, (recent - earlier) / earlier, guarded with np.where(earlier > 0, ..., NaN) to avoid divide-by-zero; NaN where a business had 0 activity in the earlier period (not treated as 0% or infinite growth).
- `*_change` — raw difference, recent - earlier, always defined including when earlier = 0 (added after growth, in a follow-up pass reloading the saved CSV).

**Rating change:** `recent_average_rating - earlier_average_rating`, per-business, NaN when a business has no reviews in one or both windows.

**Missing-value handling:** count-type columns (total_reviews, recent_reviews, earlier_reviews, total_checkins, recent_checkins, earlier_checkins) filled with 0 for businesses with no matching activity (left-join produces NaN, not a true gap); rate/change columns `(*_growth, *_average_rating, rating_change)` intentionally left as NaN rather than filled, since 0 or a placeholder would misrepresent "no data in this window."

**Validation performed:** row count assertion (8,812 in = 8,812 out), zero duplicate business_ids, and a monthly-vs-overall total cross-check on the single highest-volume business (5,778 reviews / 18,615 check-ins matched exactly between monthly and lifetime aggregates) as a sanity check on the groupby logic.

### 3.Merchant Engagement Score

**Missing `rating_change` handling**: `rating_change` is only defined for businesses with reviews in both the recent and earlier windows (3,602 of 8,812; the other 5,210 have no review in one or both periods). A separate `rating_change_for_score` column fills these gaps with `0` (neutral change) — the original `rating_change` column is left untouched for transparency. This neutral fill is scoring-only, not a claim that the rating didn't move.

**Normalization**: all three trend metrics (`review_change`, `checkin_change`, `rating_change_for_score`) are Min-Max scaled to [0, 1] via `sklearn.MinMaxScaler`, fit across all 8,812 businesses. No winsorization or outlier capping — extreme values are treated as genuine merchant activity, not noise. A neutral `rating_change_for_score` of 0 maps to ≈0.5 (midpoint) after scaling, since the observed range is roughly symmetric (-4 to +4).
**Component correlation check**: `review_trend_norm` and `checkin_trend_norm` are moderately correlated (r = 0.478); `rating_trend_norm` is essentially independent of both (r ≈ 0.01–0.02). All three retained as distinct dimensions of engagement.

**Provisional weighting**: `engagement_score` = 44.44% `review_trend_norm` + 33.33% `checkin_trend_norm` + 22.22% `rating_trend_norm` — a rescaling of the original business-specified 40:30:20:10 weighting (review/checkin/rating/sentiment) to sum to 100%, since the sentiment component doesn't exist yet at this point in the pipeline (added in a later notebook). Score range: 0.2624–0.8748, no missing values.

**Weight sensitivity check**: compared against an equal-weight (33.33/33.33/33.33) alternative by ranking all 8,812 merchants under each scheme. Median absolute rank change = 31 positions, mean = ~234, max = 6,257. Confirms the weighting materially affects individual rankings, especially for merchants with imbalanced trend profiles — the business-driven weighting was kept anyway since it reflects the intended relative importance of each signal, not because the sensitivity was small.

**Note**: this score is explicitly provisional and will be recomputed once the sentiment component is available, at which point the weighting will also be revisited.

### 4.Merchant Health Classification 

**Threshold method**: fixed Q25/Q75 cut points on `engagement_score` (0.5135836386 / 0.5198412698), computed once and applied via `pd.cut` with `(-inf, Q25]`, `(Q25, Q75]`, `(Q75, inf)` bins — rather than `pd.qcut`, which would force an exact 25/50/25 split by cutting through tied scores. Because a large number of merchants share the same score, exact quartile boundaries would arbitrarily split ties into different categories; fixed thresholds keep all tied merchants in the same bucket instead.

**Classification rule**: `Declining` if `engagement_score <= Q25`; `Stable` if `Q25 < engagement_score <= Q75`; `Growing` if `engagement_score > Q75`.

**Resulting distribution**: Declining 2,361 (26.79%), Stable 4,266 (48.41%), Growing 2,185 (24.80%) — close to but not exactly 25/50/25, as expected from the tie-preserving approach. 0 missing classifications.

**Validation performed**: confirmed the classified set contains only the three expected labels, every one of the 8,812 merchants is classified, and structural checks (shape, uniqueness, no duplicate rows) hold after reload from disk.

**Sanity check**: reviewed the top 5 merchants by engagement score within each status category to confirm the ranking behaves sensibly (e.g. high review/check-in growth merchants surfacing under Growing).