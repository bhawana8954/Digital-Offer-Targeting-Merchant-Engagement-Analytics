# Digital Offer Targeting & Merchant Engagement Analytics

An end-to-end analytics project combining **incrementality/uplift modeling** (Criteo Uplift dataset) with **merchant engagement scoring and GenAI-driven sentiment analysis** (Yelp Open Dataset), built to demonstrate data science skills relevant to digital offer targeting and merchant health monitoring.

## Project Structure

```
├── data/
│ ├── raw/ # Gitignored — original/unprocessed data pulls and intermediate checkpoints
│ └── samples/ # Committed — final, clean sample datasets used across notebooks
├── docs/ # Data dictionary, methodology notes, findings
├── notebooks/ # Analysis notebooks, organized by workstream
├── outputs/
│ ├── figures/ # Generated charts
│ ├── tables/ # Generated summary tables (CSV)
│ └── narratives/ # GenAI-generated executive narratives
├── src/ # Shared utility code (e.g. GenAI helper functions)
├── requirements.txt
└── .gitignore
```
- `docs/glossary.md` — business terminology, formulas, and statistical definitions used throughout the project

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
   Yelp Open Dataset filtering/sampling was performed on Kaggle (server-side, due to local hardware constraints with the ~5.3GB review file). Only the filtered sample outputs are used locally — see `docs/data_dictionary.md` for details.

## Workstreams

- **Criteo Uplift Modeling** — incrementality analysis and treatment-aware customer targeting model on RTB advertising data. See `notebooks/criteo_uplift/`.
- **Yelp Merchant Engagement** — merchant health scoring, GenAI sentiment analysis, and review theme extraction. See `notebooks/yelp_merchant/`.

- **Criteo Uplift Modeling** — incrementality analysis (global + segment-level) and treatment-aware customer targeting model on RTB advertising data. See `notebooks/criteo_uplift/`.
  - `01_data_preparation.ipynb` — data audit, validation, 1% dev sampling
  - `02_incrementality_analysis.ipynb` — treatment-vs-control lift (conversion & visit), full-dataset two-proportion z-test, heterogeneous treatment effect segmentation
  - `03_customer_targeting_model.ipynb` — T-Learner uplift model (v1, diagnostic — superseded by v2)
  - `04_customer_targeting_model_v2.ipynb` — T-Learner v2 (fixed): removed class balancing, added regularization tuning + feature standardization, benchmarked against a random baseline
  - `05_customer_targeting_model_evaluation.ipynb` — model performance metrics (ROC-AUC, Log Loss), percentile-based targeting groups, cumulative uplift curve, and a business targeting recommendation

*(This README is being built incrementally as each notebook is reviewed and documented — sections will expand accordingly.)*
