# Data Dictionary — Yelp Open Dataset

For the Criteo Uplift dataset dictionary, see `data_dictionary_criteo.md`.

## Raw Filtered Samples (Notebook 06 — Kaggle Sampling)

**Source:** [Yelp Open Dataset](https://www.yelp.com/dataset) (`business.json`, `review.json`, `checkin.json`), filtered on Kaggle (server-side, no local download of raw files) via `06_sampling_kaggle.ipynb`
**Filter criteria:** `categories` contains "Restaurants" (substring match) AND `city` ∈ {Philadelphia, Tampa} — the two highest-restaurant-count cities in the full dataset
**Data quality:** 0 duplicate `business_id`s in the business sample; 0 orphan `business_id`s in the review and check-in samples (every ID traces back to the business sample); 229 of the 8,812 sampled businesses have no check-in record (expected — not a data quality issue)

### `data/raw/yelp/yelp_business_sample.csv`
**Scale:** 8,812 rows × 14 columns

| Column | Type | Description |
|---|---|---|
| `business_id` | str | Unique business identifier |
| `name` | str | Business name |
| `address`, `city`, `state`, `postal_code` | str | Location fields (city restricted to Philadelphia/Tampa) |
| `latitude`, `longitude` | float | Geolocation |
| `stars` | float | Average star rating |
| `review_count` | int | Total review count (Yelp-reported, full-platform) |
| `is_open` | int (0/1) | Business open/closed status |
| `attributes` | dict/str | Business attributes (e.g. delivery, parking) |
| `categories` | str | Comma-separated category tags (filtered on "Restaurants") |
| `hours` | dict/str | Operating hours by day |

### `data/raw/yelp/yelp_review_sample_part_01.parquet` … `part_10.parquet`
**Scale:** 990,521 rows × 9 columns combined, split into 10 parts (~100K rows / ~40MB each) for Kaggle→local transfer only — recombine via `glob` + `concat` before use

| Column | Type | Description |
|---|---|---|
| `review_id` | str | Unique review identifier |
| `user_id` | str | Reviewer identifier |
| `business_id` | str | Foreign key to business sample |
| `stars` | int | Star rating (1–5) given in this review |
| `useful`, `funny`, `cool` | int | Yelp community vote counts on the review |
| `text` | str | Review text |
| `date` | datetime | Review timestamp |

### `data/raw/yelp/yelp_checkin_sample.csv`
**Scale:** 8,583 rows × 2 columns (one row per business with ≥1 check-in)

| Column | Type | Description |
|---|---|---|
| `business_id` | str | Foreign key to business sample |
| `date` | str | Comma-separated string of check-in timestamps for this business |

---

## `data/samples/yelp_merchant_engagement.csv` — after Notebook 07 (Merchant Engagement Table)

**Scale:** 8,812 rows (one per sampled business) × 28 columns. Built by joining business, review, and check-in samples and computing recent-vs-earlier (6-month) trend metrics.

| Column | Type | Description |
|---|---|---|
| *(14 original business columns)* | — | Same as `yelp_business_sample.csv` — see above |
| `total_reviews` | int | Lifetime review count for the business |
| `earlier_reviews`, `recent_reviews` | int | Review counts in the earlier (7/19/21–1/19/22 preceding 6mo) and recent (last 6mo) windows |
| `review_growth` | float | `(recent_reviews - earlier_reviews) / earlier_reviews`; `NaN` if earlier_reviews = 0 |
| `review_change` | int | `recent_reviews - earlier_reviews`; always defined, including when earlier = 0 |
| `average_rating` | float | Mean star rating across all reviews |
| `earlier_average_rating`, `recent_average_rating` | float | Mean star rating within each window; `NaN` if no reviews in that window |
| `rating_change` | float | `recent_average_rating - earlier_average_rating`; `NaN` if either window has no reviews |
| `total_checkins` | int | Lifetime check-in event count (after exploding the raw timestamp string) |
| `earlier_checkins`, `recent_checkins` | int | Check-in event counts per window |
| `checkin_growth` | float | Percent change, same zero-guard logic as `review_growth` |
| `checkin_change` | int | Raw difference, same logic as `review_change` |

## `data/samples/yelp_merchant_engagement.csv` — after Notebook 08 (Merchant Engagement Score)

**Scale:** 8,812 rows × 33 columns. Adds normalized trend scores and a composite engagement score on top of the Notebook 07 columns (updated in place).

| Column | Type | Description |
|---|---|---|
| `rating_change_for_score` | float | `rating_change` with missing values filled to `0` (neutral), used only for normalization/scoring; the original `rating_change` column is preserved as-is |
| `review_trend_norm` | float | `review_change` Min-Max scaled to [0, 1] across all businesses |
| `checkin_trend_norm` | float | `checkin_change` Min-Max scaled to [0, 1] across all businesses |
| `rating_trend_norm` | float | `rating_change_for_score` Min-Max scaled to [0, 1]; businesses with no `rating_change` land at ≈0.5 (neutral midpoint) |
| `engagement_score` | float | Weighted composite: 44.44% `review_trend_norm` + 33.33% `checkin_trend_norm` + 22.22% `rating_trend_norm`. Range 0.2624–0.8748. **Provisional** — recomputed once the sentiment component was added |

## `data/samples/yelp_merchant_engagement.csv` — after Notebook 09 (Merchant Health Classification)

**Scale:** 8,812 rows × 34 columns. Adds a merchant health classification on top of the Notebook 08 columns (updated in place).

| Column | Type | Description |
|---|---|---|
| `merchant_status` | category | `Declining`, `Stable`, or `Growing`, assigned from fixed Q25/Q75 thresholds on `engagement_score` (0.5135836386 / 0.5198412698). No missing values. |

## `data/samples/yelp_sentiment_sample.csv`

**Scale:** 2,193 rows × 11 columns. Stratified sample of reviews (up to 5 most recent per period) from 300 merchants (100 each Declining/Stable/Growing), built in Notebook 10 for downstream GenAI sentiment extraction.

| Column | Type | Description |
|---|---|---|
| `review_id` | str | Unique review identifier |
| `user_id` | str | Reviewer identifier |
| `business_id` | str | Foreign key to the merchant engagement table |
| `stars` | int | Star rating (1–5) given in this review |
| `useful`, `funny`, `cool` | int | Yelp community vote counts on the review |
| `text` | str | Review text |
| `date` | datetime | Review timestamp |
| `period` | str | `earlier` or `recent` — which 6-month window this review falls into |
| `merchant_status` | category | `Declining`, `Stable`, or `Growing` — the merchant's health classification at time of sampling |

## `data/samples/yelp_review_sentiment_final.csv`

**Scale:** 2,193 rows × 4 columns. GenAI-extracted sentiment for every review in `yelp_sentiment_sample.csv`, built in Notebook 11. Join back to `yelp_sentiment_sample.csv` on `review_id` to recover `business_id`, `stars`, `period`, `merchant_status`, etc.

| Column | Type | Description |
|---|---|---|
| `review_id` | str | Foreign key to `yelp_sentiment_sample.csv` |
| `sentiment_label` | str | `positive`, `neutral`, or `negative` |
| `sentiment_score` | float | Sentiment polarity, -1.0 (very negative) to 1.0 (very positive) |
| `sentiment_reason` | str | Concise (≤15 word) model-generated explanation for the label |

## `data/samples/yelp_review_themes_final.csv`

**Scale:** 2,193 rows × 2 columns. GenAI-extracted CX themes for every review in the sentiment sample, built in Notebook 13.

| Column | Type | Description |
|---|---|---|
| `review_id` | str | Foreign key to `yelp_sentiment_sample.csv` / `yelp_review_sentiment_final.csv` |
| `themes` | str | 1–2 comma-separated theme tags from the fixed 7-category taxonomy (see `glossary_business_terms.md`) |

## `outputs/tables/merchant_theme_summary.csv`

**Scale:** 300 rows (one per sampled merchant) × 11 columns. Per-merchant theme mention counts, built in Notebook 13.

| Column | Type | Description |
|---|---|---|
| `business_id` | str | Merchant identifier |
| `name` | str | Merchant name |
| `merchant_status` | category | `Declining`, `Stable`, or `Growing` |
| `priority_group` | str | `Expand`, `Monitor / Intervene`, `Growth Opportunity`, or `Reassess` |
| `service_speed`, `staff_behavior`, `food_product_quality`, `pricing_value`, `cleanliness_ambiance`, `order_accuracy_wait_time`, `other_none` | int | Count of this merchant's sampled reviews mentioning each theme |

## `data/samples/yelp_merchant_priority_final.csv`

**Scale:** 300 rows (one per sentiment-covered merchant) × 40 columns. Built across Notebooks 12–13: engagement metrics + merchant-level sentiment + investment-priority classification + top CX theme.

| Column | Type | Description |
|---|---|---|
| *(34 columns from the merchant engagement table)* | — | Unchanged — see `yelp_merchant_engagement.csv` above |
| `merchant_sentiment_score` | float | Mean GenAI `sentiment_score` across this merchant's sampled reviews, range [-1, 1] |
| `sentiment_review_count` | int | Number of sampled reviews this merchant's sentiment score is based on |
| `engagement_polarity` | str | `High` (merchant_status ∈ {Growing, Stable}) or `Low` (Declining) |
| `sentiment_polarity` | str | `Positive` (`merchant_sentiment_score > 0`) or `Negative` (`<= 0`) |
| `priority_group` | str | `Expand`, `Monitor / Intervene`, `Growth Opportunity`, or `Reassess` — derived from `engagement_polarity` × `sentiment_polarity` |
| `top_cx_theme` | str | This merchant's 1–2 most-mentioned CX themes (comma-separated), or `other_none` if no theme was mentioned |

## `outputs/narratives/executive_narrative_raw.md`

Unedited Groq-generated draft (model `openai/gpt-oss-120b`, `temperature=0.3`) of the executive brief, built in Notebook 15. Kept for traceability; superseded by `executive_narrative.md` as the actual deliverable.

## `outputs/narratives/executive_narrative.md`

Final, hand-polished executive brief combining Criteo incrementality findings and Yelp merchant engagement/CX findings into a 3-section narrative (ad targeting impact; merchant risk & CX drivers; strategic recommendations), built in Notebook 15 from `executive_narrative_raw.md`.