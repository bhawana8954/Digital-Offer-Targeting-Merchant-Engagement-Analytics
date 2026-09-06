# Yelp Workstream Findings: Merchant Engagement & CX Priority

**Bottom line:** Of 8,812 restaurants, ~27% are Declining and ~25% are Growing on engagement alone. Layering GenAI sentiment onto the 300-merchant sample splits that population into four actionable groups — the two "mismatch" groups (Monitor/Intervene and Growth Opportunity) need opposite interventions, and treating them the same would misallocate effort.

## Scoring Approach

Each of the 8,812 sampled restaurants (Philadelphia + Tampa) is scored on a recent-vs-earlier (6-month) engagement trend across three signals — review volume change, check-in volume change, and rating change — Min-Max normalized and combined into a weighted `engagement_score` (44.44% review / 33.33% check-in / 22.22% rating), then bucketed into `Declining` / `Stable` / `Growing` via fixed Q25/Q75 thresholds. A 300-merchant stratified sample (100 per health tier) then had its recent reviews run through a GenAI sentiment pipeline (Groq `openai/gpt-oss-20b`) to add a sentiment dimension on top of engagement. Full derivation in `methodology_yelp.md`, Sections 2–7.

## Sentiment Model Validation

GenAI sentiment score correlates strongly with the review's own star rating (Pearson r = 0.9191, Spearman rho = 0.8221, both p ≈ 0, n = 2,193 reviews) — the model's judgments broadly track human ratings rather than diverging from them. A manual review of 12 disagreement cases found genuine mixed reviews and literal readings of sarcasm, not systematic model failure.

## Key Finding — Engagement and Sentiment Measure Different Things

A merchant can be highly engaged (rising reviews/check-ins) while its customers are unhappy, or lightly engaged while its customers are delighted. Engagement alone — the metric available for all 8,812 merchants — cannot tell these two cases apart; only the sentiment-covered 300-merchant sample can. This is why the priority framework below combines both signals rather than ranking on engagement alone.

## Full-Portfolio Merchant Health (all 8,812 merchants)

| Status | Count | Share |
|---|---|---|
| Stable | 4,266 | 48.41% |
| Declining | 2,361 | 26.79% |
| Growing | 2,185 | 24.80% |

0 missing classifications.

## Investment Priority Groups (300-merchant sentiment-covered sample)

Engagement polarity (High = Growing/Stable, Low = Declining) × sentiment polarity (Positive = score > 0, Negative = score ≤ 0):

| Priority Group | Definition | Count | Share |
|---|---|---|---|
| Expand | High engagement + Positive sentiment | 147 | 49.00% |
| Growth Opportunity | Low engagement + Positive sentiment | 79 | 26.33% |
| Monitor / Intervene | High engagement + Negative sentiment | 53 | 17.67% |
| Reassess | Low engagement + Negative sentiment | 21 | 7.00% |

**Coverage caveat:** this breakdown covers only the 300 sampled merchants (3.4% of the 8,812-merchant portfolio) — the sample is stratified equally across health tiers for comparison purposes, not proportional to the true population, so these percentages should not be read as portfolio-wide priority shares.

## Top CX Drivers Behind Declining Merchants

Among Declining-tier merchants in the sample, the most-mentioned review themes were:

| Theme | Mentions | Share of Declining-tier mentions |
|---|---|---|
| Food / product quality | 597 | 45.06% |
| Staff behavior | 317 | 23.92% |
| Pricing / value | 138 | 10.42% |

## Recommendation — Differentiated Intervention Strategy

The two "mismatch" groups need opposite fixes, not the same playbook:

- **Monitor / Intervene** (high engagement, negative sentiment) — engagement is already strong, so the lever is service quality and CX. Prioritize the specific drivers above (food quality, staff behavior, pricing) rather than promotional spend. Representative examples: The Freshworks–Mayfair Store, Tasties, Boston Market.
- **Growth Opportunity** (low engagement, positive sentiment) — sentiment is already favorable, so the lever is visibility and activity, not service fixes. Prioritize promotions, loyalty programs, and repeat-visit campaigns. Representative examples: China Garden, Cask Social Kitchen, Independence Beer Garden.
- **Expand** (high engagement, positive sentiment, the largest group at 49%) — the healthiest tier; a candidate for continued investment and case-study material rather than intervention.
- **Reassess** (low engagement, negative sentiment, smallest at 7%) — weakest on both axes; warrants a closer case-by-case look before committing further spend.

## Conclusion

Engagement scoring alone is sufficient to flag which merchants are trending down across the full 8,812-merchant portfolio, but it can't say *why* — that requires the sentiment/CX-theme layer, currently available for only the 300-merchant sample. Scaling sentiment coverage to the full portfolio (or a larger stratified sample) would be the natural next step before this framework is used to allocate real intervention budget.