# Directory Structure

Full repository layout and a short description of what lives in each file/folder. For what each pipeline stage does (not just where it lives), see `methodology_criteo.md` / `methodology_yelp.md`.

```text
├── data/                          # All datasets — raw pulls and clean derived samples
│   │
│   ├── raw/                       # Gitignored — original/unprocessed data pulls and checkpoints
|   |   |
│   │   ├── criteo/
|   |   |   |
│   │   │   ├── criteo-research-uplift-v2.1.csv.gz        # Full Criteo dataset (~13.98M rows), pulled from Hugging Face
│   │   │   ├── criteo-research-uplift-v2.1-dev.csv.gz    # Dev-sized pull used during pipeline setup
│   │   │   └── .cache/huggingface/                       # Hugging Face download cache (auto-generated, gitignored)
│   │   └── yelp/
|   |       |
│   │       ├── yelp_review_sample_part_01.parquet …      # 10 row-chunked parts of the raw review sample
│   │       │   … part_10.parquet                         # (Kaggle → local transfer; recombine via glob + concat)
│   │       ├── yelp_review_sentiment_checkpoint.csv      # Resumable checkpoint from GenAI sentiment extraction
│   │       └── yelp_review_themes_checkpoint.csv         # Resumable checkpoint from GenAI theme extraction
│   │
│   └── samples/                   # Committed — final, clean datasets used across notebooks
|       |
│       ├── criteo_sample.csv                             # 1% dev sample of the Criteo dataset (random_state=42)
│       ├── customer_targeting_test_final.parquet         # Scored T-Learner v2 test set (uplift, deciles)
│       ├── yelp_business_sample.csv                      # 8,812 filtered Philadelphia/Tampa restaurants
│       ├── yelp_checkin_sample.csv                       # Check-in timestamps for the sampled businesses
│       ├── yelp_merchant_engagement.csv                  # Engagement scores + health status, all 8,812 merchants
│       ├── yelp_merchant_priority_final.csv              # 300-merchant sentiment + priority-group table
│       ├── yelp_review_sentiment_final.csv               # GenAI sentiment labels/scores, 2,193 reviews
│       ├── yelp_review_themes_final.csv                  # GenAI CX theme tags, 2,193 reviews
│       └── yelp_sentiment_sample.csv                     # Stratified 2,193-review sample fed to GenAI extraction
│
├── docs/                          # Methodology, data dictionaries, glossary, and findings
|   |
│   ├── criteo_targeting_findings.md     # Business-facing findings & recommendation — Criteo targeting
│   ├── yelp_engagement_findings.md      # Business-facing findings & recommendation — Yelp engagement
│   ├── methodology_criteo.md            # What was done and why — Criteo uplift workstream (nb 01–05)
│   ├── methodology_yelp.md              # What was done and why — Yelp engagement workstream (nb 06–15)
│   ├── data_dictionary_criteo.md        # Schema and provenance — Criteo datasets
│   ├── data_dictionary_yelp.md          # Schema and provenance — Yelp datasets
│   ├── glossary_business_terms.md       # Business/domain terminology used throughout
│   ├── glossary_formulas_stats.md       # Formulas and statistical definitions used throughout
│   └── directory_structure.md           # This file — full repository layout
│
├── notebooks/
|   |
│   │   14_final_business_recommendations.ipynb   # Cross-workstream recommendations (read-only synthesis)
│   │   15_genai_executive_narrative.ipynb        # C-suite executive brief, generated via Groq
│   │
│   ├── criteo_uplift/              # 01–05: incrementality analysis + customer targeting model
|   |   |
│   │   ├── 01_data_preparation.ipynb
│   │   ├── 02_incrementality_analysis.ipynb
│   │   ├── 03_customer_targeting_model.ipynb
│   │   ├── 04_customer_targeting_model_v2.ipynb
│   │   └── 05_customer_targeting_model_evaluation.ipynb
│   │
│   └── yelp_merchant/              # 06–13: sampling, engagement scoring, sentiment, theme extraction
|       |
│       ├── 06_sampling_kaggle.ipynb
│       ├── 07_merchant_engagement.ipynb
│       ├── 08_merchant_engagement_score.ipynb
│       ├── 09_merchant_health_classification.ipynb
│       ├── 10_genai_sentiment.ipynb
│       ├── 11_genai_sentiment_extraction.ipynb
│       ├── 12_sentiment_validation_and_priority.ipynb
│       └── 13_review_theme_extraction.ipynb
│
├── outputs/
|   |   
│   ├── figures/                    # Generated charts (PNG)
|   |   |
│   │   ├── absolute_lift.png
│   │   ├── conversion_rate_comparison.png
│   │   ├── feature_distributions.png
│   │   ├── relative_lift.png
│   │   ├── segment_incremental_lift.png
│   │   ├── segment_lift_confidence_intervals.png
│   │   ├── segment_population_vs_lift.png
│   │   ├── segment_treatment_vs_control.png
│   │   └── visit_rate_comparison.png
│   │
│   ├── narratives/                 # GenAI-generated executive briefs
|   |   |
│   │   ├── executive_narrative_raw.md      # Unedited Groq draft (traceability copy)
│   │   └── executive_narrative.md          # Final, hand-polished deliverable
│   │
│   └── tables/                     # Generated summary tables (CSV)
|       |
│       ├── customer_targeting_model_evaluation.csv
│       ├── incrementality_statistical_test.csv
│       ├── incrementality_summary.csv
│       ├── merchant_theme_summary.csv
│       └── segment_incrementality_summary.csv
│
├── src/
|   |
│   ├── genai_utils.py                # Shared GenAI utilities — batched Groq calls, structured-output
│   │                                 # validation, retry/backoff, checkpointing (used by nb 11, 13)
│   └── __pycache__/                  # Compiled bytecode cache (auto-generated, gitignored)
│
├── .gitignore
├── README.md                         # Project overview, key results, setup, and documentation guide
└── requirements.txt
```

## Notes

- **`data/raw/`** is gitignored in full — it's regenerated by re-running the download steps in the root `README.md` (Criteo via Hugging Face, Yelp via the Kaggle sampling notebook). `.cache/` and `__pycache__/` subfolders are tooling-generated and never committed.
- **`data/samples/`** is the only data folder committed to the repo — every notebook past `01`/`06` reads from here, not from `raw/`.
- **`notebooks/14` and `15`** sit at the `notebooks/` root rather than inside either workstream subfolder, since they combine Criteo and Yelp results into one cross-workstream output.
- Every file under `docs/` is referenced from the root `README.md`'s documentation table.