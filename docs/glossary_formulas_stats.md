# Glossary — Formulas & Statistical Definitions

Reference doc for the math used across the project's notebooks. For business/domain terminology, see `glossary_business_terms.md`.

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
p_value = 2 × (1 − Φ(|z|))  # Φ = standard normal CDF; implemented via scipy.stats.norm.sf
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

**T-Learner Uplift** (per customer)
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

*(This glossary will keep growing as later notebooks are reviewed.)*