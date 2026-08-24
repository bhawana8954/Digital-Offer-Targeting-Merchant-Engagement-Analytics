### Yelp Open Dataset — Sampling Methodology

**Source:** Yelp Open Dataset (via Kaggle-hosted mirror), processed in a Kaggle
Notebook rather than downloaded locally, due to the size of `review.json`
(~5.3 GB uncompressed).

**Filtering approach:**
- Category filter: `categories` contains "Restaurants"
- Geographic filter: `city` in [Philadelphia, Tampa]
- Reason: full dataset (150,346 businesses) is too large for local
  development; filtering to a category + two metro areas produces a
  manageable, representative analytical sample while preserving enough
  volume for trend and engagement analysis

**Resulting sample:**
| Metric | Value |
|---|---|
| Restaurant businesses (total) | 8,812 |
| — Philadelphia | 5,852 |
| — Tampa | 2,960 |
| Reviews (linked to sampled businesses) | 990,521 |
| Businesses represented in reviews | 8,812 / 8,812 (100%) |
| Businesses with check-in records | 8,583 / 8,812 (97.4%) |
| Businesses with no check-in record | 229 |
| Orphan business IDs (reviews) | 0 |
| Orphan business IDs (check-ins) | 0 |

**Data quality note:** 229 businesses (2.6%) have no check-in activity
recorded. This is treated as a legitimate "zero engagement" signal rather
than missing data, and is handled explicitly in the merchant engagement
scoring step rather than dropped.

### Merchant Engagement Trends (Step 12)

Two raw trend columns were added to the merchant engagement dataset,
alongside the existing percent-change growth metrics from Step 11:

| Column | Definition |
|---|---|
| `review_change` | `recent_reviews - earlier_reviews` (raw difference) |
| `checkin_change` | `recent_checkins - earlier_checkins` (raw difference) |

These are kept **separate** from the Step 11 percent-change columns
(`review_growth`, `checkin_growth`) rather than replacing them, since
the two measure different things:

- **Percent change** (`review_growth`, `checkin_growth`) — proportional
  growth; sensitive to low baselines (e.g. 1 → 21 checkins reads as a
  huge percentage even though the absolute activity is still small)
- **Raw change** (`review_change`, `checkin_change`) — absolute
  difference; better reflects real-world engagement volume, less
  distorted by small denominators

Both are retained so downstream analysis (segmentation, ranking,
visualization) can choose the metric appropriate to the question being
asked, rather than being locked into one definition of "trend."

**Output:** merchant engagement table, 8,812 rows × 28 columns
(business_id remains unique — no duplication introduced).