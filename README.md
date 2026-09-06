# Digital Offer Targeting & Merchant Engagement Analytics

An end-to-end analytics project combining **incrementality/uplift modeling** (Criteo Uplift dataset) with **merchant engagement scoring and GenAI-driven sentiment analysis** (Yelp Open Dataset), built to demonstrate data science skills relevant to digital offer targeting and merchant health monitoring.

## Key Results at a Glance

**Criteo — ad targeting incrementality**
- Full-dataset lift: control 0.194% → treatment 0.309% conversion (+59.4% relative lift, p ≈ 7.3e-179)
- 4 statistically-attractive customer segments identified (features `f0`, `f2`, `f6`, `f8`)
- T-Learner uplift model: top-decile customers capture 65% of test-set conversions; **Top 10% targeting beats random by ~4×**, benefit fades past Top 30%

**Yelp — merchant engagement & CX**
- 8,812 restaurants scored: 48.4% Stable, 26.8% Declining, 24.8% Growing
- GenAI sentiment (2,193 reviews, 300 merchants) agrees strongly with star ratings (Pearson r = 0.92)
- Priority framework splits merchants into 4 actionable groups (Expand / Monitor–Intervene / Growth Opportunity / Reassess), each with a distinct recommended action

Full detail: [`docs/criteo_targeting_findings.md`](docs/criteo_targeting_findings.md) and [`docs/yelp_engagement_findings.md`](docs/yelp_engagement_findings.md).

## Project Structure

```
├── data/
│   ├── raw/              # Gitignored — original/unprocessed data pulls and intermediate checkpoints
│   └── samples/          # Committed — final, clean sample datasets used across notebooks
├── docs/                 # Data dictionaries, methodology notes, findings, glossary
├── notebooks/
│   ├── criteo_uplift/    # 01–05
│   └── yelp_merchant/    # 06–13
│   (14, 15 combine both workstreams; stay at notebooks/ root)
├── outputs/
│   ├── figures/          # Generated charts
│   ├── tables/           # Generated summary tables (CSV)
│   └── narratives/       # GenAI-generated executive narratives
├── src/                  # Shared utility code (e.g. GenAI helper functions)
├── requirements.txt
└── .gitignore
```

## Documentation

| Doc | Covers |
|---|---|
| [`docs/methodology_criteo.md`](docs\methodology_crieto.md) | What was done and why — Criteo uplift workstream (notebooks 01–05) |
| [`docs/methodology_yelp.md`](docs\methodology_yelp.md) | What was done and why — Yelp engagement workstream (notebooks 06–15) |
| [`docs/data_dictionary_criteo.md`](docs\data_dictionary_criteo.md) | Schema and provenance for Criteo datasets |
| [`docs/data_dictionary_yelp.md`](docs\data_dictionary_yelp.md) | Schema and provenance for Yelp datasets |
| [`docs/criteo_targeting_findings.md`](docs\criteo_targeting_findings.md) | Business-facing findings & recommendation — Criteo customer targeting |
| [`docs/yelp_engagement_findings.md`](docs\yelp_engagement_findings.md) | Business-facing findings & recommendation — Yelp merchant engagement |
| [`docs/glossary_business_terms.md`](docs\glossary_business_terms.md) | Business/domain terminology used throughout |
| [`docs/glossary_formulas_stats.md`](docs\glossary_formulas_stats.md) | Formulas and statistical definitions used throughout |
| [`docs/directory_structure.md`](docs\directory_structure.md) | Full repository layout — every file/folder, described |

## Setup

1. **Clone the repo and create an isolated environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Note: `statsmodels` is intentionally excluded — no Python 3.14 build was available at the time of setup. The one function needed from it (`proportions_ztest`) was reimplemented manually using `scipy.stats`.

3. **Environment variables**

   Create a `.env` file for any API keys (e.g. Groq API key used for GenAI sentiment/theme extraction). This file is gitignored.

4. **Download the Criteo Uplift dataset**

   The original Criteo AI Lab download URL is no longer available, so the dataset is pulled programmatically from Hugging Face:
   ```bash
   python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='criteo/criteo-uplift', filename='criteo-research-uplift-v2.1.csv.gz', repo_type='dataset', local_dir='./data/raw/criteo')"
   ```
   This downloads ~13.98M rows (~311MB compressed) directly into `data/raw/criteo/`.

5. **Yelp data**

   Yelp Open Dataset filtering/sampling was performed on Kaggle (server-side, due to local hardware constraints with the ~5.3GB review file). Only the filtered sample outputs are used locally — see `docs/data_dictionary_yelp.md` for details.

## Workstreams

**Criteo Uplift Modeling** — incrementality analysis (global + segment-level) and treatment-aware customer targeting model on RTB advertising data. See `notebooks/criteo_uplift/`.
- `01_data_preparation.ipynb` — data audit, validation, 1% dev sampling
- `02_incrementality_analysis.ipynb` — treatment-vs-control lift (conversion & visit), full-dataset two-proportion z-test, heterogeneous treatment effect segmentation
- `03_customer_targeting_model.ipynb` — T-Learner uplift model (v1, diagnostic — superseded by v2)
- `04_customer_targeting_model_v2.ipynb` — T-Learner v2 (fixed): removed class balancing, added regularization tuning + feature standardization, benchmarked against a random baseline
- `05_customer_targeting_model_evaluation.ipynb` — model performance metrics (ROC-AUC, Log Loss), percentile-based targeting groups, cumulative uplift curve, and a business targeting recommendation

**Yelp Merchant Engagement** — merchant health scoring, GenAI sentiment analysis, and review theme extraction. See `notebooks/yelp_merchant/`.
- `06_sampling_kaggle.ipynb` — Yelp Open Dataset sampling/filtering, executed on Kaggle (server-side data, no local download of raw multi-GB files). Filters to Restaurants category businesses in Philadelphia + Tampa, then pulls matching reviews and check-ins.
- `07_merchant_engagement.ipynb` — builds the core merchant engagement table by combining business, review, and check-in samples: review/check-in volume and rating trends over a recent-vs-earlier 6-month window. Output: `data/samples/yelp_merchant_engagement.csv`.
- `08_merchant_engagement_score.ipynb` — normalizes review/check-in/rating trend metrics to 0–1 and combines them into a provisional weighted merchant engagement score. Updates `data/samples/yelp_merchant_engagement.csv` in place.
- `09_merchant_health_classification.ipynb` — classifies each merchant into Declining/Stable/Growing based on fixed engagement-score quartile thresholds. Updates `data/samples/yelp_merchant_engagement.csv` in place.
- `10_yelp_genai_sentiment.ipynb` — selects a stratified sample of 300 merchants (100 each Declining/Stable/Growing) and their most recent reviews per period, for downstream GenAI sentiment analysis. Output: `data/samples/yelp_sentiment_sample.csv`.
- `11_genai_sentiment_extraction.ipynb` — runs the 2,193-review sentiment sample through Groq's `openai/gpt-oss-20b` via `src/genai_utils.py`, producing per-review sentiment labels/scores/reasons. Output: `data/samples/yelp_review_sentiment_final.csv`.
- `12_sentiment_validation_and_priority.ipynb` — validates GenAI sentiment against Yelp star ratings, aggregates sentiment to merchant level, and combines it with engagement scoring into four investment-priority groups (Expand / Monitor–Intervene / Growth Opportunity / Reassess). Output: `data/samples/yelp_merchant_priority_final.csv`.
- `13_review_theme_extraction.ipynb` — extracts CX themes (service speed, staff behavior, food quality, etc.) from the same review sample via Groq, builds a per-merchant theme-frequency table, and appends each merchant's top theme(s) to the priority table. Outputs: `data/samples/yelp_review_themes_final.csv`, `outputs/tables/merchant_theme_summary.csv`; updates `data/samples/yelp_merchant_priority_final.csv` in place.
- `14_final_business_recommendations.ipynb` — consolidates Criteo targeting results (notebooks 02–05) and Yelp merchant engagement/priority results (notebooks 06–13) into a single cross-workstream set of business recommendations. Read-only analysis — produces no new data files.
- `15_genai_executive_narrative.ipynb` — generates a C-suite executive brief synthesizing both workstreams via Groq. Outputs: `outputs/narratives/executive_narrative_raw.md` (unedited model draft), `outputs/narratives/executive_narrative.md` (final, hand-polished brief).

**Shared utilities**
- `src/genai_utils.py` — shared GenAI utility module: review compression, batched Groq calls with Pydantic-validated structured output, retry/backoff with item-level fallback, and CSV checkpointing for resumable long-running extraction jobs. Used by `11_genai_sentiment_extraction.ipynb` (and later theme-extraction work).

## Tech Stack

Pure Python — `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, and Groq (via `src/genai_utils.py`) for the GenAI sentiment/theme/narrative steps. No SQL layer, by design (kept separate from this candidate's existing SQL/Power BI portfolio work).

