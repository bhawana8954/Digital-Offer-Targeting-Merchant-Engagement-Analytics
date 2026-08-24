## **Workstream 1 Findings: Customer Targeting Model**

*Model approach:* Built a T-learner uplift model using two separate regularized logistic regressions (treatment-group model and control-group model, `C=0.01`, standardized features), after diagnosing and correcting severe miscalibration in an initial naive version (removing `class_weight="balanced"`, adding regularization, standardizing features).

*Individual model performance:* Treatment model ROC-AUC 0.9468, control model ROC-AUC 0.9400 — strong within-group conversion ranking, though based on very few positive examples (52 treatment / 27 control training conversions; 13 treatment / 7 control test conversions), so these AUCs should be read as encouraging but not statistically robust.

*Key finding — AUC vs. uplift are different things:* Strong individual-model AUC did not translate into a cleanly monotonic uplift ranking. This is expected, not contradictory: uplift is a difference of two independently-noisy probability estimates, which amplifies rather than cancels their errors — a known limitation of T-learners specifically.

*Targeting group results:*
| Group | Population % | Predicted Uplift | Observed Uplift |
|---|---|---|---|
| Top 10% | 10% | 0.0072 | 0.0057 |
| 10–25% | 15% | 0.0010 | -0.0001 |
| 25–50% | 25% | 0.0005 | -0.0000 |
| Bottom 50% | 50% | 0.0002 | 0.0015* |

*Bottom 50%'s positive observed uplift is a sparsity artifact (zero control conversions occurred in that half of the test set), not evidence of a real reversed effect.

*Cumulative targeting comparison (vs. random baseline uplift of 0.0014):*
- Top 10%: 0.0057 (meaningfully above random)
- Top 20%: 0.0028 (above random, weaker)
- Top 30%: 0.0012 (roughly at/below random — benefit doesn't sustain)

*Conclusion:* Targeting the highest predicted-uplift customers produces a real, measurable improvement over random/blanket targeting, concentrated mainly in the top 10–20% of the population. The model's ranking is not reliable across the full population (non-monotonic mid-to-low bands), and the top decile's edge — while real (65% of test conversions fall in the top predicted decile) — rests on a small number of total conversions (20), so results should be presented as an exploratory, directionally useful targeting signal rather than a production-ready incremental-response model.

*Recommendation:* Prioritize the Top 10% for promotional exposure; Top 20% is a reasonable expansion if broader reach is needed; targeting beyond Top 30% shows no evidence of incremental benefit over random targeting.