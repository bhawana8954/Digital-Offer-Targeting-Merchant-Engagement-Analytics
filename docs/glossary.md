# Glossary: Business Terms, Formulas & Statistical Definitions

Reference doc for terminology and math used across the project's notebooks. Updated incrementally as each notebook is reviewed.

## Business / Domain Terms

| Term | Definition |
|---|---|
| **Uplift / Incrementality** | The *causal* effect of a treatment (e.g. showing an ad) on an outcome — the difference between what happened with treatment and what would have happened without it, isolated via a control group. Distinct from a raw conversion rate, which conflates people who'd have converted anyway. |
| **RTB (Real-Time Bidding)** | The auction-based mechanism by which digital ad impressions are bought/sold in real time. |
| **Treatment (ITT — Intent-to-Treat)** | Whether a customer was *assigned* to receive the offer (e.g. a bid was placed), regardless of whether the ad actually reached them. Used as the primary variable for causal analysis because it preserves the randomization — unlike `exposure`. |
| **Exposure** | Whether the customer *actually* received the treatment (auction won, ad rendered). Conditioning on exposure introduces selection bias, since auction wins aren't random. |
| **Conversion** | The target business outcome (e.g. a purchase). |
| **Visit** | An intermediate/secondary outcome (e.g. site visit), often more frequent than conversion and useful when conversion is very sparse. |
| **T-Learner** | An uplift-modeling approach that trains two separate models — one on the treatment group, one on the control group — and estimates individual uplift as the difference between their predicted outcome probabilities for the same customer. |
| **Uplift Decile Ranking** | A model evaluation method: rank customers by predicted uplift, bucket into 10 equal-sized groups (deciles), and compare each decile's *predicted* uplift to its *observed* (actual) uplift. A well-calibrated model shows observed uplift decreasing smoothly from decile 1 (highest predicted) to decile 10 (lowest) — similar in spirit to a Qini curve.
| **Attractive Segment** | A customer segment (this project's term) that passes three screening criteria simultaneously: above-average lift, adequate sample size, and statistical significance. See Statistical Definitions below. |
| **Conversion Concentration** | The share of total observed conversions captured within a given top-ranked slice of customers (e.g. "top decile captures 65% of conversions") — a practical way to judge targeting value when absolute uplift estimates are unreliable due to sparse data. |
| **Random Baseline (Uplift Modeling)** | A non-predictive benchmark created by randomly ranking customers into deciles instead of using model scores. A real uplift model should be judged against this baseline, not against a theoretical zero, since even random ranking shows some decile-to-decile fluctuation from sampling noise. |
| **Cumulative Uplift Curve** | A chart showing observed incremental lift as more customers are added to the targeted population, ordered by predicted uplift (highest first). Used to identify the point of diminishing returns for a targeting campaign — conceptually similar to a Qini curve. |
| **Percentile-Based Targeting Group** | Grouping customers into wide bands by their rank in predicted uplift (e.g. Top 10%, Bottom 50%) rather than fine-grained deciles, to get larger, more statistically stable samples per group — a practical response to conversion sparsity. |
| **Orphan ID** | A business_id appearing in a filtered dataset (reviews or check-ins) that does not exist in the business sample. Used as a referential-integrity check after filtering; a non-zero count would indicate a filtering bug.|
| **Category substring filter** | Filtering businesses by checking whether a target string (e.g. "Restaurants") appears anywhere within the comma-separated categories field, rather than requiring an exact category match. |
| **Recent / earlier period** | The two fixed 6-month comparison windows used throughout the Yelp engagement analysis, anchored to the dataset's max review date (2022-01-19): recent = last 6 months, earlier = the 6 months before that. Not relative to the analysis run date. |
| **Growth vs. change** | Two ways of expressing a recent-vs-earlier trend: growth is a percent change (NaN when the earlier period had zero activity, to avoid a misleading 0%/∞ result); change is the raw difference (always defined, including from a zero base). |
| **Neutral fill (scoring)** | Replacing a missing trend value with a midpoint-equivalent value (e.g. 0 before normalization, ≈0.5 after) so a business with insufficient data neither helps nor hurts its composite score, rather than being excluded or defaulted to the worst case. Used for `rating_change_for_score` / `rating_trend_norm` in the engagement score. |
| **Weight sensitivity check** | Comparing rankings produced under the actual scoring weights against an equal-weight alternative, to gauge how much the chosen weighting scheme (versus the underlying data) drives the final ranking. |
| **Fixed-threshold binning vs. `qcut`** | Classifying values against pre-computed quantile cut points (`pd.cut` with fixed bin edges) rather than using `pd.qcut`, which recomputes bin edges to force an exact percentage split. Fixed thresholds keep merchants with identical scores in the same category rather than splitting ties across categories to hit an exact 25/50/25 distribution.

## Formulas

**Conversion / Visit Rate** (within a group)
```text
rate = mean(outcome) = conversions / total_customers
```

**Absolute Lift**
```text
absolute_lift = treatment_rate − control_rate
```

(percentage-point difference)

**Relative Lift**
```text
relative_lift = (treatment_rate − control_rate) / control_rate
```

(treatment improvement as a % of the control baseline)

**Pooled Conversion Rate** (for hypothesis testing)
```text
p_pooled = (treatment_conversions + control_conversions) / (treatment_total + control_total)
```

**Pooled Standard Error** (two-proportion z-test, under H₀: rates are equal)
```text
SE_pooled = sqrt( p_pooled × (1 − p_pooled) × (1/n_treatment + 1/n_control) )
```

**Z-Statistic**
```text
z = (treatment_rate − control_rate) / SE_pooled
```

**Two-Sided P-Value**
```text
p_value = 2 × (1 − Φ(|z|)) # Φ = standard normal CDF; implemented via scipy.stats.norm.sf
```

**Unpooled Standard Error** (for the confidence interval — different from the SE used in the hypothesis test)
```text
SE_unpooled = sqrt( p_t(1−p_t)/n_t + p_c(1−p_c)/n_c )
```

**95% Confidence Interval** (on the rate difference)
```text
margin_of_error = 1.96 × SE_unpooled
CI = (rate_difference − margin_of_error, rate_difference + margin_of_error)
```

**T-Learner Uplift (per customer)**
```text
uplift = P(convert | X, T=1) − P(convert | X, T=0)
```
where the two probabilities come from separately trained treatment-arm and control-arm models.

## Statistical Definitions

| Term | Definition |
|---|---|
| **Null Hypothesis (H₀)** | The default assumption of no effect — here, that treatment and control conversion rates are equal. |
| **Alternative Hypothesis (H₁)** | What's tested against H₀ — here, that the rates differ (two-sided). |
| **Significance Level (α)** | The threshold probability (0.05 in this project) below which a result is called "statistically significant" — i.e. the accepted rate of false positives. |
| **P-Value** | The probability of observing a difference this large (or larger) if H₀ were actually true. p < α → reject H₀. |
| **Two-Proportion Z-Test** | A hypothesis test comparing two independent binomial proportions (here, conversion rates of two groups), assuming large sample size (normal approximation to the binomial). |
| **Confidence Interval (95% CI)** | A range that would contain the true rate difference in 95% of repeated samples. If it excludes zero, the difference is significant at α = 0.05 — consistent with (but computed differently from) the p-value check. |
| **Pooled vs. Unpooled Standard Error** | Pooled SE assumes H₀ is true (equal rates) and is used *for the hypothesis test*; unpooled SE makes no such assumption and is the statistically correct choice *for the confidence interval*. |
| **Multiple Testing Risk** | Running many significance tests (e.g. one per customer segment) inflates the overall false-positive rate beyond the nominal α for any single test. Segment-level findings in this project are flagged as exploratory for this reason. |
| **Class Imbalance** | When one outcome class (e.g. conversion=1) is far rarer than the other. Severe imbalance (here, ~0.3% conversion rate) makes model training and evaluation unstable, especially in small subgroups. |
| **Quasi-Complete Separation** | A logistic regression failure mode where the model can (near-)perfectly separate classes using the available features, usually due to few positive cases and/or high dimensionality relative to sample size — leads to extreme, overconfident, poorly calibrated predicted probabilities. |
| **Stratified Sampling / Splitting** | Sampling or splitting data such that the proportions of a key variable (or combination of variables, e.g. treatment×conversion) are preserved in every subset — used here to keep rare conversions represented in both train and test sets. |
| **Downsampling** | Reducing the size of a majority group to match a minority group, to avoid a model learning to just predict the majority class. |
| **Regularization (L2, parameter C in scikit-learn)** | A penalty on model coefficient size that discourages overfitting. In scikit-learn's `LogisticRegression`, `C` is the *inverse* of regularization strength — smaller `C` = stronger penalty = simpler, less extreme predictions. Useful when a model (like the v1 T-Learner) overfits to a small number of rare positive cases. |
| **Feature Standardization** | Rescaling features to zero mean and unit variance (`StandardScaler`) before fitting a regularized model. Necessary because regularization penalizes coefficient size uniformly — unscaled features with different ranges would be penalized unevenly. |
| **ROC-AUC** | Measures a model's ability to rank positive cases (converters) above negative cases across all possible classification thresholds, without committing to one threshold. Ranges 0.5 (random) to 1.0 (perfect ranking). |
| **Log Loss** | Measures the quality of predicted *probabilities* (not just rankings) — penalizes confident-but-wrong predictions heavily. Lower is better. Less interpretable in isolation when the outcome is very rare. |

*(This glossary will keep growing as later notebooks — segmentation refinements, v2 modeling, Yelp engagement/sentiment work — are reviewed.)*