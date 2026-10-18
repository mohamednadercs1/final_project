"""Step 1 - Ingest the raw TMDB 5000 CSVs, parse JSON-like columns, treat zeros as missing.

Run:  python data_cleaning.py
Out:  data/movies_parsed.pkl  (one row per film; list columns already parsed)
"""
import argparse
import ast
import json
import re

import numpy as np
import pandas as pd

import config

SEQUEL_TITLE_RE = re.compile(r"(?:\b[2-9]\b|\b(?:II|III|IV)\b|\bPart\b)")


# ----------------------------------------------------------------- JSON parsing
def parse_json_list(value):
    """Safely parse a JSON / Python-literal string into a list; return [] on any failure."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            out = loader(value)
            return out if isinstance(out, list) else []
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
    return []


def extract_names(value, n=None):
    """Return the 'name' of each dict in a JSON list column (optionally the first n)."""
    names = [d.get("name") for d in parse_json_list(value) if isinstance(d, dict) and d.get("name")]
    return names[:n] if n else names


def extract_director(crew):
    for d in parse_json_list(crew):
        if isinstance(d, dict) and d.get("job") == "Director" and d.get("name"):
            return d["name"]
    return None


def extract_top_cast(cast, n=3):
    """Top-n billed actors, ordered by TMDB billing order."""
    people = [d for d in parse_json_list(cast) if isinstance(d, dict) and d.get("name")]
    people.sort(key=lambda d: d.get("order", 999))
    return [d["name"] for d in people[:n]]


# ----------------------------------------------------------------- loading & cleaning
def load_raw(movies_path=config.RAW_MOVIES, credits_path=config.RAW_CREDITS):
    """Load and merge the two Kaggle files (credits.movie_id == movies.id)."""
    movies = pd.read_csv(movies_path).drop_duplicates("id")
    credits = (pd.read_csv(credits_path)
               .rename(columns={"movie_id": "id"})
               .drop(columns=["title"], errors="ignore")
               .drop_duplicates("id"))
    return movies.merge(credits, on="id", how="inner")


def parse_dataframe(raw):
    """Parse nested columns, set zero numerics to NaN, keep released films only."""
    df = raw.copy()
    if "status" in df:
        df = df[df["status"].eq("Released")]
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    for col in ("budget", "revenue", "runtime"):
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)  # 0 means "missing"

    df["genres_list"] = df["genres"].map(extract_names)
    df["studios_list"] = df["production_companies"].map(extract_names)
    df["keywords_list"] = df["keywords"].map(lambda x: [k.lower() for k in extract_names(x)])
    df["director"] = df["crew"].map(extract_director)
    df["cast_top3"] = df["cast"].map(extract_top_cast)
    df["is_sequel"] = (df["keywords_list"].map(lambda k: "sequel" in k)
                       | df["title"].astype(str).str.contains(SEQUEL_TITLE_RE)).astype(int)

    keep = ["id", "title", "release_date", "budget", "revenue", "runtime", "original_language",
            "popularity", "vote_average", "vote_count",   # post-release: kept for history features only
            "genres_list", "studios_list", "keywords_list", "director", "cast_top3", "is_sequel"]
    return df[keep].reset_index(drop=True)


def filter_modelable(df, verbose=True):
    """Drop rows that cannot support a trustworthy target (missing / implausible money columns).

    - budget or revenue still missing (the target is never imputed)
    - budget/revenue below a plausibility floor (placeholder values such as $1)
    - suspected foreign-currency entries: non-English films with budget >= 1e9
      or budget >= 50x revenue
    """
    n0 = len(df)
    d = df.dropna(subset=["release_date", "budget", "revenue"])
    d = d[(d["budget"] >= config.MIN_BUDGET) & (d["revenue"] >= config.MIN_REVENUE)]
    foreign = d["original_language"].ne("en")
    suspect = foreign & ((d["budget"] >= 1e9) | (d["budget"] / d["revenue"] >= 50))
    d = d[~suspect].copy()
    if verbose:
        print(f"filter_modelable: {n0} -> {len(d)} rows "
              f"({int(suspect.sum())} flagged as suspected foreign-currency entries)")
    return d


def zero_report(df):
    return pd.Series({
        "rows": len(df),
        "budget missing/zero": int(df["budget"].isna().sum()),
        "revenue missing/zero": int(df["revenue"].isna().sum()),
        "runtime missing/zero": int(df["runtime"].isna().sum()),
        "no director": int(df["director"].isna().sum()),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--movies", default=config.RAW_MOVIES)
    ap.add_argument("--credits", default=config.RAW_CREDITS)
    ap.add_argument("--out", default=config.PARSED)
    args = ap.parse_args()

    df = parse_dataframe(load_raw(args.movies, args.credits))
    print(zero_report(df).to_string())
    df.to_pickle(args.out)
    print(f"Saved {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
