"""Step 2 - Enrich films with the TMDB v3 API (/movie/{id}).

Fetches imdb_id, US certification (MPAA), updated budget / revenue / runtime / release date and
production companies. Zero or missing budget / revenue / runtime in the Kaggle file are CORRECTED
with TMDB's current value when TMDB has one (observed data only - never statistically imputed).

Vote average / vote count / popularity are saved with a `post_` prefix for reference only: they are
measured after release and driven by success, so no model ever uses them (look-ahead bias).

Credentials (either one, in .env or the environment):
    TMDB_API_KEY      v3 API key           (sent as ?api_key=...)
    TMDB_READ_TOKEN   v4 read access token (sent as Authorization: Bearer ...)

Run (resumable - every response is cached on disk):
    python tmdb_enrichment.py
    python tmdb_enrichment.py --limit 200
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

import config

BASE_URL = "https://api.themoviedb.org/3"


class RateLimiter:
    """Allow at most `per_second` calls per second (TMDB tolerates roughly 40-50)."""

    def __init__(self, per_second):
        self.min_interval = 1.0 / per_second
        self._last = 0.0

    def wait(self):
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


class TMDBClient:
    """Small TMDB client with headers, rate limiting, retries/backoff and an on-disk cache."""

    def __init__(self, credential, cache_dir, requests_per_second=20, max_retries=4):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "movie-success-project/1.0 (educational use)",
        })
        self.params = {}
        if credential.startswith("eyJ"):                     # v4 read token (a JWT)
            self.session.headers["Authorization"] = f"Bearer {credential}"
        else:                                                # v3 API key
            self.params["api_key"] = credential
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(requests_per_second)
        self.max_retries = max_retries

    def get_movie(self, movie_id):
        """Return the movie JSON ({"_missing": True} on 404) or None on a transient failure."""
        cache_file = self.cache_dir / f"{movie_id}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        params = {**self.params, "append_to_response": "release_dates,external_ids"}
        data = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                r = self.session.get(f"{BASE_URL}/movie/{movie_id}", params=params, timeout=15)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                data = {"_missing": True}
                break
            if r.status_code == 401:
                sys.exit("TMDB rejected the credentials (HTTP 401). Check TMDB_API_KEY / TMDB_READ_TOKEN.")
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 2)) + 1)
                continue
            if r.ok:
                data = r.json()
                break
            time.sleep(2 ** attempt)

        if data is None:
            return None          # not cached -> retried on the next run
        cache_file.write_text(json.dumps(data))
        return data


def us_certification(js):
    """First non-empty US certification (e.g. 'PG-13', 'R') from the release_dates block."""
    for country in (js.get("release_dates") or {}).get("results", []):
        if country.get("iso_3166_1") == "US":
            for rel in country.get("release_dates", []):
                if rel.get("certification"):
                    return rel["certification"]
    return None


def extract_fields(js):
    """Flatten the TMDB response into the columns we keep."""
    if not js or js.get("_missing"):
        return {}
    companies = js.get("production_companies") or []
    return {
        "imdb_id": js.get("imdb_id") or (js.get("external_ids") or {}).get("imdb_id") or None,
        "tmdb_budget": js.get("budget") or np.nan,
        "tmdb_revenue": js.get("revenue") or np.nan,
        "tmdb_runtime": js.get("runtime") or np.nan,
        "tmdb_release_date": js.get("release_date") or None,
        "us_certification": us_certification(js),
        "tmdb_n_companies": len(companies),
        "tmdb_company_names": "|".join(c.get("name", "") for c in companies),
        # post-release metrics: reference only, never model inputs
        "post_vote_average": js.get("vote_average"),
        "post_vote_count": js.get("vote_count"),
        "post_popularity": js.get("popularity"),
    }


def apply_corrections(df, enriched):
    """Fill missing budget / revenue / runtime / release_date from TMDB; returns (df, report)."""
    out = df.merge(enriched, on="id", how="left")
    out["tmdb_release_date"] = pd.to_datetime(out.get("tmdb_release_date"), errors="coerce")
    report = {}
    for col, src in [("budget", "tmdb_budget"), ("revenue", "tmdb_revenue"),
                     ("runtime", "tmdb_runtime"), ("release_date", "tmdb_release_date")]:
        if src not in out:
            report[col] = 0
            continue
        fill = out[col].isna() & out[src].notna()
        out.loc[fill, col] = out.loc[fill, src]
        report[col] = int(fill.sum())
    return out, report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parsed", default=config.PARSED)
    ap.add_argument("--out", default=config.TMDB_ENRICHED)
    ap.add_argument("--limit", type=int, default=None, help="process at most N films")
    ap.add_argument("--rps", type=float, default=20, help="requests per second")
    args = ap.parse_args()

    credential = os.getenv("TMDB_READ_TOKEN") or os.getenv("TMDB_API_KEY")
    if not credential:
        sys.exit("Missing TMDB credentials. Put TMDB_API_KEY (or TMDB_READ_TOKEN) in .env - see .env.example.")

    movies = pd.read_pickle(args.parsed)[["id"]]
    if args.limit:
        movies = movies.head(args.limit)

    client = TMDBClient(credential, config.CACHE_DIR / "tmdb", requests_per_second=args.rps)
    rows, failed = [], 0
    for i, movie_id in enumerate(movies["id"], 1):
        js = client.get_movie(int(movie_id))
        if js is None:
            failed += 1
        else:
            rows.append({"id": movie_id, **extract_fields(js)})
        if i % 100 == 0:
            print(f"{i}/{len(movies)} processed")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} rows -> {args.out} ({failed} transient failures; re-run to retry them)")
    if len(out):
        print(out[["imdb_id", "tmdb_budget", "tmdb_revenue", "us_certification"]].notna().mean().round(2))


if __name__ == "__main__":
    main()
