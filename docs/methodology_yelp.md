# Methodology — Yelp Merchant Engagement Workstream

Covers notebooks `06_sampling_kaggle.ipynb` through `15_genai_executive_narrative.ipynb`. For the Criteo uplift workstream, see `methodology_criteo.md`.

## 1. Yelp Sampling

**Execution environment:** run as a Kaggle Notebook against the Yelp Open Dataset mounted server-side — avoids downloading the ~5.3GB `review.json` locally.

**City/category selection:** businesses filtered where `categories` contains "Restaurants" (case-insensitive substring match, not exact-category equality) → 52,268 restaurant businesses. Ranked by restaurant count per city; Philadelphia (5,852) and Tampa (2,960) selected as the top two cities.

**Business sample:** 8,812 businesses, 0 duplicate `business_id`s.

**Review sample:** `review.json` streamed in 100k-row chunks (memory constraint — file too large to load in one pass) and matched against the `business_id` set incrementally. Result: 990,521 reviews, all 8,812 sampled businesses represented.

**Checkin sample:** filtered directly (small file, no chunking needed). Result: 8,583 businesses with check-in history — 229 of the 8,812 sampled businesses have no check-in record at all (expected gap, not a data quality issue).

**Referential integrity checks:** post-filter, verified 0 orphan `business_id`s in both the review sample and the check-in sample (i.e., every ID in those samples traces back to the business sample).

**Kaggle → local transfer:** the review sample (400MB+ as a single parquet) wouldn't reliably download as one file from the Kaggle UI, so it was split into 10 row-chunked parquet parts (~100k rows / ~40MB each) for transfer, to be recombined locally via `glob` + `concat`. Business and check-in samples were small enough to download directly as CSV.

## 2. Merchant Engagement Table

**Inputs combined:** business sample (8,812 rows), review sample (10 parquet parts reloaded via `glob` + `concat`, 990,521 rows), check-in sample (8,583 rows). All joined on `business_id`, left-joined onto the business table so every one of the 8,812 businesses is retained.

**Recent vs. earlier window:** defined relative to the max review date in the dataset (2022-01-19), not the run date — recent = last 6 months (2021-07-19 to 2022-01-19), earlier = the preceding 6 months (2021-01-19 to 2021-07-19). Reviews/check-ins outside both windows are labeled "outside" and excluded from trend calculations (used only for lifetime totals).

**Check-in event expansion:** raw check-in rows store a single comma-separated string of all timestamps per business. Before aggregation, this string is split and exploded into one row per check-in event, then parsed to datetime (unparseable timestamps dropped) — 1,796,822 individual check-in events recovered from 8,583 businesses.

**Growth vs. change:** two parallel metrics computed for reviews and check-ins:
- `*_growth` — percent change, `(recent - earlier) / earlier`, guarded with `np.where(earlier > 0, ..., NaN)` to avoid divide-by-zero; `NaN` where a business had 0 activity in the earlier period (not treated as 0% or infinite growth).
- `*_change` — raw difference, `recent - earlier`, always defined including when earlier = 0 (added after growth, in a follow-up pass reloading the saved CSV).

**Rating change:** `recent_average_rating - earlier_average_rating`, per-business, `NaN` when a business has no reviews in one or both windows.

**Missing-value handling:** count-type columns (`total_reviews`, `recent_reviews`, `earlier_reviews`, `total_checkins`, `recent_checkins`, `earlier_checkins`) filled with 0 for businesses with no matching activity (left-join produces `NaN`, not a true gap); rate/change columns (`*_growth`, `*_average_rating`, `rating_change`) intentionally left as `NaN` rather than filled, since 0 or a placeholder would misrepresent "no data in this window."

**Validation performed:** row count assertion (8,812 in = 8,812 out), zero duplicate `business_id`s, and a monthly-vs-overall total cross-check on the single highest-volume business (5,778 reviews / 18,615 check-ins matched exactly between monthly and lifetime aggregates) as a sanity check on the groupby logic.

## 3. Merchant Engagement Score

**Missing `rating_change` handling:** `rating_change` is only defined for businesses with reviews in both the recent and earlier windows (3,602 of 8,812; the other 5,210 have no review in one or both periods). A separate `rating_change_for_score` column fills these gaps with `0` (neutral change) — the original `rating_change` column is left untouched for transparency. This neutral fill is scoring-only, not a claim that the rating didn't move.

**Normalization:** all three trend metrics (`review_change`, `checkin_change`, `rating_change_for_score`) are Min-Max scaled to [0, 1] via `sklearn.MinMaxScaler`, fit across all 8,812 businesses. No winsorization or outlier capping — extreme values are treated as genuine merchant activity, not noise. A neutral `rating_change_for_score` of 0 maps to ≈0.5 (midpoint) after scaling, since the observed range is roughly symmetric (-4 to +4).

**Component correlation check:** `review_trend_norm` and `checkin_trend_norm` are moderately correlated (r = 0.478); `rating_trend_norm` is essentially independent of both (r ≈ 0.01–0.02). All three retained as distinct dimensions of engagement.

**Provisional weighting:** `engagement_score` = 44.44% `review_trend_norm` + 33.33% `checkin_trend_norm` + 22.22% `rating_trend_norm` — a rescaling of the original business-specified 40:30:20:10 weighting (review/checkin/rating/sentiment) to sum to 100%, since the sentiment component doesn't exist yet at this point in the pipeline (added in Section 5). Score range: 0.2624–0.8748, no missing values.

**Weight sensitivity check:** compared against an equal-weight (33.33/33.33/33.33) alternative by ranking all 8,812 merchants under each scheme. Median absolute rank change = 31 positions, mean = ~234, max = 6,257. Confirms the weighting materially affects individual rankings, especially for merchants with imbalanced trend profiles — the business-driven weighting was kept anyway since it reflects the intended relative importance of each signal, not because the sensitivity was small.

**Note:** this score is explicitly provisional and was recomputed once the sentiment component became available (see `yelp_engagement_findings.md`), at which point the weighting was also revisited.

## 4. Merchant Health Classification

**Threshold method:** fixed Q25/Q75 cut points on `engagement_score` (0.5135836386 / 0.5198412698), computed once and applied via `pd.cut` with `(-inf, Q25]`, `(Q25, Q75]`, `(Q75, inf)` bins — rather than `pd.qcut`, which would force an exact 25/50/25 split by cutting through tied scores. Because a large number of merchants share the same score, exact quartile boundaries would arbitrarily split ties into different categories; fixed thresholds keep all tied merchants in the same bucket instead.

**Classification rule:** `Declining` if `engagement_score <= Q25`; `Stable` if `Q25 < engagement_score <= Q75`; `Growing` if `engagement_score > Q75`.

**Resulting distribution:** Declining 2,361 (26.79%), Stable 4,266 (48.41%), Growing 2,185 (24.80%) — close to but not exactly 25/50/25, as expected from the tie-preserving approach. 0 missing classifications.

**Validation performed:** confirmed the classified set contains only the three expected labels, every one of the 8,812 merchants is classified, and structural checks (shape, uniqueness, no duplicate rows) hold after reload from disk.

**Sanity check:** reviewed the top 5 merchants by engagement score within each status category to confirm the ranking behaves sensibly (e.g. high review/check-in growth merchants surfacing under Growing).

## 5. Sentiment Review Sampling

**Eligibility:** a merchant qualifies for sentiment sampling only if it has at least one review in both the recent and earlier 6-month windows (same windows as the engagement analysis). Of 8,812 merchants, 3,602 are eligible (1,754 Declining, 1,531 Growing, 317 Stable) — eligibility varies sharply by status because Growing/Declining merchants are defined by recent activity, while Stable merchants by definition have less review volume in general.

**Stratified sampling, not proportional:** exactly 100 merchants sampled from each status category (300 total, `random_state=42`), rather than sampling proportionally to the population. This is intentional — the goal is to compare sentiment *across* merchant health categories, not to estimate population-level sentiment, so equal representation of each category matters more than mirroring the natural distribution.

**Review cap per merchant-period:** each sampled merchant contributes up to 5 reviews per period, sorted by date and taking the most recent — not a random selection — since the goal is to characterize sentiment as of that period, and the most recent reviews within a window are the closest representation of it. Merchants with fewer than 5 eligible reviews contribute all they have. The 5-review cap was chosen because the median eligible merchant has ~5 reviews per period; requiring 5 in both periods for every merchant would have excluded 175 of the 300 selected merchants.

**Result:** 2,193 reviews total (1,093 earlier, 1,100 recent), all 300 sampled merchants represented in both periods, 0 duplicate review IDs, no merchant exceeding 5 reviews in either period.

**Scope note:** this sample supports comparative merchant-level sentiment analysis (Declining vs. Stable vs. Growing), not population-level sentiment estimation — the equal stratification breaks the natural population proportions by design.

## 6. GenAI Sentiment Extraction

**Model & determinism:** Groq-hosted `openai/gpt-oss-20b`, `temperature=0.0` to keep sentiment classification as consistent/reproducible as possible across runs.

**Text compression before sending to the model:** reviews over 120 words are compressed via `compress_review()` — kept in full if ≤120 words or ≤3 sentences (up to the 120-word cap in that case); otherwise reduced to the first 2 and last 2 sentences, to preserve opening/closing context while cutting token usage on long reviews (max observed: 894 words).

**Structured output enforcement:** each batch prompt requires a strict JSON schema (`review_id`, `sentiment_label` ∈ {positive, neutral, negative}, `sentiment_score` ∈ [-1.0, 1.0], `sentiment_reason` ≤15 words), validated against a Pydantic model (`SingleReviewSentiment`) on return — invalid labels, out-of-range scores, or missing `review_id`s in the response raise an error rather than being silently accepted.

**Evidence-based prompting:** the prompt explicitly instructs the model to base sentiment only on what's stated in the review text, not to infer unstated causes (e.g. a negative review isn't labeled a price complaint unless price is actually mentioned).

**Batching & rate limits:** reviews processed in batches of 5 per API call. The full 2,193-review run used a reduced batch size of 2 (down from the batch size of 5 used in the initial sanity test) to work within Groq API rate limits.

**Retry & fallback:** each batch retries up to 3 times with capped exponential backoff (max 10s between attempts). If a batch still fails after all retries, it's split and retried item-by-item; any single review that still can't be parsed gets a fallback record (`neutral`, `0.0`, reason `"Fallback assigned due to parsing error"`) rather than being dropped, so one bad review never blocks the whole run.

**Checkpointing:** results are written to a checkpoint CSV after every batch, so an interrupted run (rate limit, crash) resumes from the last completed batch instead of restarting — the notebook shows exactly this behavior, resuming with all 2,193 already completed on a later rerun.

**Post-processing before production save:** the checkpoint is cleaned of any leftover sanity-test rows (`review_id` starting with `test_`) and any fallback rows before being copied to the final production file — the run in question produced 0 fallback rows, so no records were actually dropped this time, but the cleanup step is a standing safeguard.

**Result:** 2,193 reviews scored, 0 duplicates, all labels within the valid set, all scores within bounds. Sentiment distribution: positive 1,455 (66.3%), negative 627 (28.6%), neutral 111 (5.1%).

## 7. Sentiment Validation & Merchant Priority

**Validation approach:** sentiment output re-merged with source review metadata (`validate="one_to_one"` on every merge, explicit row-count and null-coverage assertions) rather than assumed correct — catches silent join errors (duplicated or dropped rows) immediately.

**Sentiment-vs-rating agreement:** Pearson r = 0.9191, Spearman rho = 0.8221 (both p ≈ 0) across all 2,193 reviews — strong positive association between GenAI sentiment score and Yelp star rating, confirming the sentiment model is broadly consistent with human ratings. Both correlations were marginally lower than an earlier partial-data run (r = 0.9270, rho = 0.8320), a negligible shift attributed to sample completion rather than a change in model behavior.

**Disagreement review:** 12 reviews (6 five-star/negative-sentiment, 6 one-or-two-star/positive-sentiment) manually inspected. Findings: some are sarcasm the model read literally (e.g. an exaggerated "highly recommended" after graphic negative imagery), others are genuine mixed reviews (positive about the food, negative about a specific issue like delivery) — not systematic model failures.

**Merchant-level aggregation:** sentiment averaged per `business_id` (mean `sentiment_score`, count of reviews) for the 300 sampled merchants; reconciled to confirm the summed review counts equal the full 2,193-review set.

**Coverage gap handling:** sentiment covers only the 300 sampled merchants (3.40% of all 8,812). Missing sentiment is left as `NaN`, explicitly not filled with `0` — `0` means "measured as neutral," `NaN` means "not measured." Engagement score and merchant health status remain available for all 8,812 merchants regardless of sentiment coverage; only sentiment-dependent analyses are restricted to the 300.

**Engagement × sentiment framework:** for sentiment-covered merchants only — engagement polarity is High (Growing/Stable) or Low (Declining); sentiment polarity is Positive (`score > 0`) or Negative (`score <= 0`, ties classified negative to keep the split mutually exclusive and binary).

**Priority groups:** four groups from the 2×2 combination — Expand (High+Positive, 147/49.00%), Monitor/Intervene (High+Negative, 53/17.67%), Growth Opportunity (Low+Positive, 79/26.33%), Reassess (Low+Negative, 21/7.00%). Compared against an earlier partial-data run, Expand's share grew (+7.9pp) and Monitor/Intervene's shrank (-7.8pp) on the completed dataset — a meaningful shift, flagged rather than glossed over.

**Qualitative spot-check:** 2–3 merchants sampled per priority group (`random_state=42`) and manually reviewed against their underlying engagement/sentiment scores, as a sanity check — not a statistical validation — that the rule-based grouping produces sensible assignments.

## 8. Review Theme Extraction

**Closed taxonomy:** reviews classified into 1–2 tags from a fixed 7-category list (`service_speed`, `staff_behavior`, `food_product_quality`, `pricing_value`, `cleanliness_ambiance`, `order_accuracy_wait_time`, `other_none`) — the model is explicitly instructed not to invent new tags, with `other_none` as the required fallback when nothing else fits.

**Model & parsing:** same Groq `openai/gpt-oss-20b` at `temperature=0.0` as sentiment extraction, batches of 10, up to 3 retries per batch for malformed/invalid JSON responses (parsed via a JSON-extraction helper that strips markdown code fences before parsing).

**Data quality fixes applied post-run:** one stray `review_id` present in the checkpoint but absent from the 2,193-review source sample was identified and removed before final save — this was the source of an earlier 2,194-vs-2,193 row mismatch. A hard validation check (raising an error if any survive) now confirms zero invalid `"none"` theme tags remain in the final output; the checkpoint-resume logic also treats any previously-saved row with an invalid theme as unprocessed, so a future rerun would auto-retry rather than requiring manual cleanup.

**Theme aggregation:** each review's theme tags exploded into individual (business, theme) pairs, pivoted into a per-merchant theme-count matrix, and merged onto the 300-merchant priority table's metadata (name, status, priority group).

**Top CX drivers (Declining merchants):** `food_product_quality` (597 mentions, 45.06% of all Declining-tier theme mentions), `staff_behavior` (317, 23.92%), `pricing_value` (138, 10.42%) — the three most-cited issues among merchants already flagged as declining.

**Top theme per merchant:** for each of the 300 merchants, the 1–2 most-mentioned themes (mentions > 0) are joined into a `top_cx_theme` field; merchants with zero mentions of any theme default to `"other_none"`. Merged onto the priority table (`validate="one_to_one"`), all 300 merchants covered, 0 missing.

## 9. Final Business Recommendations & Executive Narrative

**Cross-workstream reference:** Notebook 14 re-displays and re-derives the established Criteo findings (overall incrementality, the 4 attractive segments, T-Learner evaluation, percentile targeting groups) directly from their saved output tables — included so the notebook can be read standalone without reopening earlier ones.

**Absolute vs. relative lift ranking divergence:** the 4 attractive Criteo segments are ranked separately by absolute lift and by relative lift, and the top segment differs between the two orderings (`f8` leads on absolute lift; `f6` leads on relative lift) — both rankings are shown rather than picking one, since which matters more depends on whether the business goal is maximizing total incremental conversions or maximizing response rate within a segment.

**Full-portfolio merchant health:** all 8,812 merchants classified — Stable 48.41%, Declining 26.79%, Growing 24.80% (0 missing).

**Priority-sample recommendations:** within the 300-merchant sentiment-covered sample — Expand 49.00%, Growth Opportunity 26.33%, Monitor/Intervene 17.67%, Reassess 7.00%.

**Differentiated intervention strategy (the notebook's core recommendation):** Monitor/Intervene merchants (high engagement, negative sentiment) should get service-quality/CX-focused attention, since their engagement is already strong but sentiment is dragging value down; Growth Opportunity merchants (low engagement, positive sentiment) should get engagement-focused investment (promotions, loyalty, repeat-visit campaigns), since sentiment is already favorable and the gap is activity volume. Treating both groups the same way would misallocate effort — one needs a fix, the other needs fuel. Representative examples cited per group (e.g. The Freshworks–Mayfair Store, Tasties, Boston Market for Monitor/Intervene; China Garden, Cask Social Kitchen, Independence Beer Garden for Growth Opportunity) to ground the abstract engagement/sentiment scores in named merchants.

**Unified cross-workstream payload:** a single `executive_summary_data` structure assembles Criteo's control/treatment conversion rates, relative lift, p-value, and top attractive segment alongside Yelp's `priority_group` tier counts and top-3 Declining-merchant CX themes (computed over all 7 taxonomy categories, matching Section 8's denominator) — one payload feeding one narrative, rather than two separate write-ups.

**Two-stage generation:** (1) a raw draft generated by Groq `openai/gpt-oss-120b` (a larger model than the `20b` used for sentiment/theme extraction, given this is a fluency-driven writing task rather than structured classification) at `temperature=0.3` — saved unedited to `executive_narrative_raw.md` for traceability; (2) a hand-polished `cleaned_narrative`, manually rewritten from the raw draft by the author, saved as the final deliverable `executive_narrative.md`. Keeping both means the unedited model output stays auditable even after manual editing.

**Narrative structure:** three sections — Criteo ad incrementality & precision targeting impact, Yelp merchant risk & operational CX drivers, and strategic recommendations & action plan — explicitly designed to read as a single cross-workstream C-suite brief rather than two disconnected summaries.

For the business-facing summary of this workstream, see `yelp_engagement_findings.md`.