# Movie Commercial Success Prediction 🎬

Predicts whether a film will be commercially successful from its characteristics, TMDB metadata and
external critic reception, and tests whether critic scores add predictive value beyond what is known
at greenlight stage.

## Pipeline

```
data/tmdb_5000_*.csv ─► data_cleaning.py ─► tmdb_enrichment.py ─► scraper.py ─► features.py ─► train_models.py ─► app.py
      (Kaggle)            parse JSON,          TMDB API v3:         OMDb API +      history +       4 models, CV +      Streamlit
                          zeros → NaN          imdb_id, MPAA,       optional RT     seasonality     temporal test,      demo
                                               fill zero values     scrape          features        SHAP, costs
```

| File | Purpose |
|---|---|
| `config.py` | Paths, thresholds, target definitions, business costs, `.env` loader |
| `data_cleaning.py` | Merge Kaggle files, parse JSON-like columns, zero → missing, plausibility filters |
| `tmdb_enrichment.py` | TMDB `/movie/{id}` client (rate limit, retries, cache); corrects zero budget/revenue/runtime; MPAA rating |
| `scraper.py` | Critic scores via OMDb (primary) and optional Rotten Tomatoes scrape (robots.txt-aware fallback) |
| `features.py` | Targets, leakage-safe director/cast/studio history, seasonality, genre/studio encoding |
| `train_models.py` | LR / Random Forest / LightGBM / XGBoost, stratified CV + temporal test, cost-benefit matrix, SHAP |
| `app.py` | Streamlit UI: budget, genres, cast, critic score → hit/flop probability |
| `dev/make_sample_data.py` | Synthetic Kaggle-format data for smoke tests (results are meaningless) |
| `tests/` | Unit tests (`pytest -q`) |
| `final_project.ipynb` | Exploratory notebook version of the analysis |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your keys (Windows: copy .env.example .env)
```

1. Download the **TMDB 5000 Movie Dataset** from Kaggle into `data/` (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`).
2. Free keys: TMDB (v3 key or v4 read token) and OMDb (activate it via the email link).

## Run

```bash
python data_cleaning.py          # -> data/movies_parsed.pkl
python tmdb_enrichment.py        # -> data/tmdb_enriched.csv      (~4,800 requests, cached/resumable)
python scraper.py --limit 900    # -> data/critic_scores.csv      (OMDb free tier = 1,000/day; re-run daily)
python features.py               # -> data/modeling_frame.pkl
python train_models.py           # ROI target; also: --target revenue | --target tier
streamlit run app.py
```

Steps 2 and 3 are optional for a first run: without them you get a greenlight-only model from the
Kaggle data. No keys at all? Smoke-test with synthetic data:
`python dev/make_sample_data.py --n 2500 --extras`.

## Design decisions

- **Targets:** `success_roi` = revenue ≥ 2.5× budget (default), `success_rev` = revenue ≥ $100M,
  `tier` = Low (<1×) / Medium / High (≥2.5×). Revenue is never imputed.
- **Look-ahead bias:** `popularity`, `vote_average`, `vote_count`, IMDb rating and RT *audience*
  score are measured after release and are excluded from every model (enforced by an assertion in
  `train_models.py`). Director/cast/studio history uses only films released *earlier*.
  Only critic scores (Tomatometer / Metascore) are used. Scores fetched today are still a snapshot,
  so the critic model is "post-screening", not strictly pre-release.
- **Validation:** stratified 5-fold CV on the training period for model/threshold selection;
  the most recent 20% of films is a temporal test set that is never used for tuning.
- **Metrics:** PR-AUC (primary, compare with the base rate), ROC-AUC, F1, precision, recall, and a
  cost-benefit confusion matrix (`--cost-fp`, `--cost-fn`, in $M).
- **Critic value:** greenlight vs. +critic on identical films with a paired-bootstrap 95% CI.
- **Deployed model:** the better of LightGBM / XGBoost by CV PR-AUC, refit on all films.

## Known limitations

- Nominal (not inflation-adjusted) dollars; revenue is box office only.
- Films with missing budget/revenue are dropped, biasing toward well-documented releases.
- TMDB's current figures largely match the Kaggle snapshot, so API corrections fill only some zeros.
- Rotten Tomatoes' terms discourage scraping and its markup changes; the scraper fallback is
  best-effort and off by default. Prefer OMDb.
- The test set is a few hundred films, so metric differences are noisy; read the bootstrap CIs.
