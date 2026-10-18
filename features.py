"""Step 4 - Feature engineering.

* build_modeling_frame(): merge parsed data + TMDB corrections + critic scores, filter, add the three
  target definitions and leakage-safe "creative power" history features.
* FeatureBuilder: fit on TRAIN only (genre vocabulary, top studios, genre budget medians), then
  transform any frame into model-ready numeric features.
* CareerLookup: career statistics used to score a brand-new film (Streamlit app).

Run:  python features.py   ->  data/modeling_frame.pkl
"""
import re

import numpy as np
import pandas as pd

import config
from data_cleaning import filter_modelable
from tmdb_enrichment import apply_corrections

VALUES = ["log_roi", "log_rev", "vote"]
GROUPS = ["director", "cast", "studio"]
HISTORY_COLS = [f"{g}_prior_{v}" for g in GROUPS for v in VALUES] + [f"{g}_n_prior" for g in GROUPS]
CERTS = ["G", "PG", "PG-13", "R"]


# ------------------------------------------------------------------ targets
def add_targets(df):
    d = df.copy()
    d["roi"] = d["revenue"] / d["budget"]
    d["success_rev"] = (d["revenue"] >= config.REVENUE_THRESHOLD).astype(int)
    d["success_roi"] = (d["roi"] >= config.ROI_THRESHOLD).astype(int)
    lo, hi = config.ROI_TIERS
    d["tier"] = np.select([d["roi"] < lo, d["roi"] < hi], [0, 1], default=2).astype(int)
    d["log_roi"] = np.log(d["roi"]).clip(-3, 5)
    d["log_rev"] = np.log(d["revenue"])
    d["vote"] = d["vote_average"].where(d["vote_average"] > 0)
    return d


# ------------------------------------------------------------------ creative-power history
def _long_table(df, group):
    """One row per (film, person) for a group of people (director / top-3 cast / studios)."""
    col = {"director": "director", "cast": "cast_top3", "studio": "studios_list"}[group]
    base = df[["release_date"] + VALUES].assign(movie_idx=df.index, person=df[col])
    if group != "director":
        base = base.explode("person")
    return base.dropna(subset=["person"])


def add_history_features(df):
    """Mean log-ROI / log-revenue / vote of each person's (or studio's) EARLIER films only.

    Rows are ordered by release date and every statistic is shift()-ed before the expanding mean,
    so a film never sees its own outcome (nor that of any later film).
    """
    d = df.reset_index(drop=True)
    for group in GROUPS:
        long = _long_table(d, group).sort_values(["release_date", "movie_idx"], kind="mergesort")
        g = long.groupby("person", sort=False)
        cols = []
        for v in VALUES:
            name = f"{group}_prior_{v}"
            long[name] = g[v].transform(lambda s: s.shift().expanding().mean())
            cols.append(name)
        long[f"{group}_n_prior"] = g.cumcount()
        cols.append(f"{group}_n_prior")
        agg = long.groupby("movie_idx")[cols].mean()
        d = d.join(agg)
        d[f"{group}_n_prior"] = d[f"{group}_n_prior"].fillna(0)
    return d


class CareerLookup:
    """Career statistics over ALL known films, for scoring a film released after the dataset ends."""

    def fit(self, df):
        d = df.reset_index(drop=True)
        self.stats_, self.counts_ = {}, {}
        for group in GROUPS:
            long = _long_table(d, group)
            self.stats_[group] = long.groupby("person")[VALUES].mean()
            self.counts_[group] = long.groupby("person").size()
        self.director_names = self.counts_["director"][self.counts_["director"] >= 2].index.sort_values().tolist()
        self.actor_names = self.counts_["cast"][self.counts_["cast"] >= 3].index.sort_values().tolist()
        self.studio_names = self.counts_["studio"][self.counts_["studio"] >= 5].index.sort_values().tolist()
        return self

    def _group(self, group, people):
        people = [p for p in people if p]
        stats = [self.stats_[group].loc[p] for p in people if p in self.stats_[group].index]
        counts = [int(self.counts_[group].get(p, 0)) for p in people]
        out = {f"{group}_prior_{v}": (float(np.mean([s[v] for s in stats if pd.notna(s[v])]))
                                       if any(pd.notna(s[v]) for s in stats) else np.nan) for v in VALUES}
        out[f"{group}_n_prior"] = float(np.mean(counts)) if counts else 0.0
        return out

    def history_row(self, director, cast, studios):
        row = {}
        row.update(self._group("director", [director] if director else []))
        row.update(self._group("cast", list(cast)))
        row.update(self._group("studio", list(studios)))
        return row


# ------------------------------------------------------------------ calendar helpers
def _thanksgiving(year):
    first = pd.Timestamp(year=int(year), month=11, day=1)
    return first + pd.Timedelta(days=(3 - first.dayofweek) % 7) + pd.Timedelta(weeks=3)


def _safe(prefix, name):
    return re.sub(r"[^0-9A-Za-z]+", "_", f"{prefix}_{name}").strip("_")


# ------------------------------------------------------------------ feature builder
class FeatureBuilder:
    """Fit on train, transform anything. Output columns are plain ASCII (LightGBM/XGBoost safe)."""

    def __init__(self, n_top_studios=25, min_genre_films=20):
        self.n_top_studios = n_top_studios
        self.min_genre_films = min_genre_films

    def fit(self, train):
        genre_counts = pd.Series([g for l in train["genres_list"] for g in l]).value_counts()
        self.genres_ = sorted(genre_counts[genre_counts >= self.min_genre_films].index)
        studio_counts = pd.Series([s for l in train["studios_list"] for s in l]).value_counts()
        self.top_studios_ = studio_counts.head(self.n_top_studios).index.tolist()
        self.genre_budget_median_ = {
            g: float(train.loc[train["genres_list"].map(lambda l: g in l), "budget"].median())
            for g in self.genres_}
        self.global_budget_median_ = float(train["budget"].median())
        self.use_cert_ = "us_certification" in train and train["us_certification"].notna().any()
        cols = list(self.transform(train).columns)
        self.greenlight_features_ = [c for c in cols if c != "critic_score"]
        return self

    def _budget_to_genre(self, budget, genres):
        meds = [self.genre_budget_median_[g] for g in genres if g in self.genre_budget_median_]
        return budget / (np.mean(meds) if meds else self.global_budget_median_)

    def transform(self, df):
        dates = pd.to_datetime(df["release_date"])
        month, dow = dates.dt.month, dates.dt.dayofweek
        thanks = dates.map(lambda t: _thanksgiving(t.year))
        days_to_tg = (dates - thanks).dt.days
        studios = df["studios_list"]

        c = {
            "log_budget": np.log1p(df["budget"].astype(float)),
            "runtime": df["runtime"].astype(float),
            "release_month": month,
            "release_quarter": dates.dt.quarter,
            "release_dow": dow,
            "is_weekend_release": dow.isin([4, 5, 6]).astype(int),   # Fri-Sun opening
            "is_summer_blockbuster": (month.isin([5, 6, 7]) | ((month == 8) & (dates.dt.day <= 15))).astype(int),
            "is_thanksgiving_window": days_to_tg.between(-7, 5).astype(int),
            "is_christmas_window": ((month == 12) & (dates.dt.day >= 10)).astype(int),
            "is_valentine_window": ((month == 2) & dates.dt.day.between(5, 18)).astype(int),
            "is_english": (df["original_language"] == "en").astype(int),
            "is_sequel": df["is_sequel"].astype(int),
            "n_genres": df["genres_list"].map(len),
            "n_studios": studios.map(len),
            "is_major_studio": studios.map(lambda l: int(any(t in s.lower() for s in l for t in config.MAJOR_STUDIO_TOKENS))),
            "log_budget_to_genre_ratio": np.log(np.array(
                [self._budget_to_genre(b, g) for b, g in zip(df["budget"], df["genres_list"])], dtype=float)),
        }
        c["is_holiday_window"] = ((c["is_thanksgiving_window"] + c["is_christmas_window"]
                                   + c["is_valentine_window"]) > 0).astype(int)
        for g in self.genres_:
            c[_safe("genre", g)] = df["genres_list"].map(lambda l, g=g: int(g in l))
        for s in self.top_studios_:
            c[_safe("studio", s)] = studios.map(lambda l, s=s: int(s in l))
        if self.use_cert_:
            for cert in CERTS:
                cert_col = df["us_certification"] if "us_certification" in df else pd.Series(None, index=df.index)
                c[_safe("cert", cert)] = (cert_col == cert).astype(int)
        for h in HISTORY_COLS:
            c[h] = df[h] if h in df else (0.0 if h.endswith("n_prior") else np.nan)
        if "critic_score" in df:
            c["critic_score"] = df["critic_score"].astype(float)

        X = pd.DataFrame(c, index=df.index)
        return X.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------ pipeline entry points
def temporal_split(df, test_fraction=config.TEST_FRACTION):
    """Chronological split: older films train, the most recent `test_fraction` test."""
    d = df.sort_values(["release_date", "id"]).reset_index(drop=True)
    cut = int(len(d) * (1 - test_fraction))
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()


def build_modeling_frame():
    df = pd.read_pickle(config.PARSED)
    if config.TMDB_ENRICHED.exists():
        df, fixed = apply_corrections(df, pd.read_csv(config.TMDB_ENRICHED))
        print("TMDB corrections applied (values filled):", fixed)
    else:
        print("note: no tmdb_enriched.csv found - skipping API corrections / MPAA ratings")
    if config.CRITIC_SCORES.exists():
        critics = pd.read_csv(config.CRITIC_SCORES)[["id", "rt_critic", "metacritic"]]
        df = df.merge(critics, on="id", how="left")
        df["critic_score"] = df[["rt_critic", "metacritic"]].mean(axis=1)   # critic scores only
        print(f"critic coverage: {df['critic_score'].notna().mean():.0%}")
    else:
        print("note: no critic_scores.csv found - post-screening model will be skipped")

    df = filter_modelable(df)
    df = add_targets(df)
    df = df.sort_values(["release_date", "id"]).reset_index(drop=True)
    return add_history_features(df)


def main():
    df = build_modeling_frame()
    df.to_pickle(config.MODELING_FRAME)
    train, test = temporal_split(df)
    fb = FeatureBuilder().fit(train)
    X = fb.transform(train)
    print(f"Saved {len(df)} films -> {config.MODELING_FRAME}")
    print(f"train {len(train)} ({train['release_date'].min().date()} .. {train['release_date'].max().date()}) | "
          f"test {len(test)} ({test['release_date'].min().date()} .. {test['release_date'].max().date()})")
    print(f"{len(fb.greenlight_features_)} greenlight features"
          f"{' + critic_score' if 'critic_score' in X else ''}")
    print("success rates:", df[["success_roi", "success_rev"]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
