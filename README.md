### Yelp Merchant Engagement Workstream

Because the full Yelp Open Dataset is several GB (review.json alone is
~5.3 GB), business/review/check-in filtering was performed server-side
in a Kaggle Notebook rather than downloaded locally. Only the filtered,
category- and city-restricted samples are included in this repo.

- Kaggle notebook: `notebooks/06-yelp-sampling-kaggle-ipynb.ipynb`
- Filter: Restaurants category, Philadelphia + Tampa metro areas
- Output samples: `data/samples/yelp_business_sample.csv`,
  `yelp_review_sample.csv`, `yelp_checkin_sample.csv`
- See `docs/data_dictionary.md` for full sampling methodology and
  resulting sample sizes.

  ### Merchant Engagement Trends (Step 12)

Trend columns (`review_change`, `checkin_change`, and rating trend) were
added to the merchant engagement dataset, measuring raw period-over-period
difference (recent minus earlier) for each metric. See
`docs/data_dictionary.md` for full definitions and how these differ from
the percent-change growth columns from Step 11.