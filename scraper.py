"""Step 3 - External critic reception: OMDb API (primary) + optional Rotten Tomatoes scrape (fallback).

Leakage policy (look-ahead bias)
--------------------------------
  CRITIC  (allowed in the post-screening model):  rt_critic (Tomatometer), metacritic (Metascore)
  AUDIENCE (never used as model inputs):          post_rt_audience, post_imdb_rating
Audience scores accumulate after the theatrical run and are driven by it. Note that critic scores
fetched today are still a *snapshot*, not the value on opening day, so treat the critic model as
"post-screening" rather than strictly pre-release.

Sources
  * OMDb API (needs OMDB_API_KEY; free tier = 1,000 requests/day, so use --limit and re-run daily).
    Matched by imdb_id, with a release-year guard against false joins.
  * Rotten Tomatoes HTML (--rt-fallback, off by default). Fetched only if robots.txt allows it, at
    a slow polite rate, and accepted only when the page's title matches. RT's markup changes and its
    terms discourage scraping, so this is best-effort: parsers are defensive and the OMDb route is
    the recommended one.

Run:
    python scraper.py --limit 900
    python scraper.py --limit 900 --rt-fallback
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.robotparser

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import config
from data_cleaning import filter_modelable
from tmdb_enrichment import RateLimiter, apply_corrections

OMDB_URL = "https://www.omdbapi.com/"
RT_BASE = "https://www.rottentomatoes.com"
USER_AGENT = "movie-success-project/1.0 (educational use; contact: see README)"


class DailyLimitReached(Exception):
    """Raised when OMDb reports its request limit."""


def _num(value):
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- OMDb
def parse_omdb(data, release_year):
    """Extract critic scores from an OMDb payload; reject mismatched years (false joins)."""
    out = {"rt_critic": None, "metacritic": None, "post_imdb_rating": None, "year_match": None}
    if not data or data.get("Response") == "False":
        return out
    try:
        out["year_match"] = abs(int(str(data.get("Year", ""))[:4]) - int(release_year)) <= 1
    except (TypeError, ValueError):
        out["year_match"] = False
    if not out["year_match"]:
        return out
    for item in data.get("Ratings", []):
        src, val = item.get("Source"), item.get("Value", "")
        if src == "Rotten Tomatoes":
            out["rt_critic"] = _num(val)
        elif src == "Metacritic":
            out["metacritic"] = _num(val.split("/")[0])
    if out["metacritic"] is None:
        out["metacritic"] = _num(data.get("Metascore"))
    out["post_imdb_rating"] = _num(data.get("imdbRating"))
    return out


def fetch_omdb(session, limiter, api_key, imdb_id, cache_dir):
    """Return (payload, made_network_call). Raises DailyLimitReached when OMDb says so."""
    cache_file = cache_dir / f"{imdb_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text()), False
    for attempt in range(3):
        limiter.wait()
        try:
            r = session.get(OMDB_URL, params={"apikey": api_key, "i": imdb_id}, timeout=15)
            payload = r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 ** attempt)
            continue
        if "limit" in str(payload.get("Error", "")).lower():
            raise DailyLimitReached()
        if r.status_code in (200, 404):
            cache_file.write_text(json.dumps(payload))
            return payload, True
        time.sleep(2 ** attempt)
    return None, True


# ---------------------------------------------------------------- Rotten Tomatoes (fallback)
def slugify(title):
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def parse_rt_html(html, title):
    """Best-effort extraction of Tomatometer / Audience score; returns {} if the title doesn't match."""
    soup = BeautifulSoup(html, "html.parser")
    out, name_ok = {}, False

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            ld = json.loads(tag.string or "")
        except ValueError:
            continue
        for item in (ld if isinstance(ld, list) else [ld]):
            if isinstance(item, dict) and item.get("@type") == "Movie":
                name_ok = _norm(item.get("name")) == _norm(title)
                out["rt_critic"] = _num((item.get("aggregateRating") or {}).get("ratingValue"))
    if not name_ok:
        return {}

    card = soup.find("script", id="media-scorecard-json")
    if card is not None and card.string:
        try:
            data = json.loads(card.string)
            out["rt_critic"] = _num((data.get("criticsScore") or {}).get("score")) or out.get("rt_critic")
            out["post_rt_audience"] = _num((data.get("audienceScore") or {}).get("score"))
        except ValueError:
            pass
    return out


class RTScraper:
    """Polite scraper: robots.txt check, identifiable User-Agent, slow rate, on-disk cache."""

    def __init__(self, cache_dir, seconds_between=3.0):
        self.cache_dir = cache_dir
        self.limiter = RateLimiter(1.0 / seconds_between)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
        self.robots = urllib.robotparser.RobotFileParser(f"{RT_BASE}/robots.txt")
        try:
            self.robots.read()
            self.allowed = True
        except Exception:
            self.allowed = False     # cannot verify robots.txt -> do not scrape

    def scrape(self, movie_id, title, year):
        cache_file = self.cache_dir / f"{movie_id}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        if not self.allowed:
            return {}
        result = {}
        for slug in (f"{slugify(title)}_{year}", slugify(title)):
            url = f"{RT_BASE}/m/{slug}"
            if not self.robots.can_fetch(USER_AGENT, url):
                continue
            self.limiter.wait()
            try:
                r = self.session.get(url, timeout=15)
            except requests.RequestException:
                continue
            if r.ok:
                result = parse_rt_html(r.text, title)
                if result:
                    break
        cache_file.write_text(json.dumps(result))
        return result


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="stop after N new OMDb requests (free tier: 1000/day)")
    ap.add_argument("--rt-fallback", action="store_true", help="scrape Rotten Tomatoes for films OMDb couldn't score")
    ap.add_argument("--omdb-rps", type=float, default=5.0)
    args = ap.parse_args()

    omdb_key = os.getenv("OMDB_API_KEY")
    if not omdb_key:
        sys.exit("Missing OMDB_API_KEY. Put it in .env - see .env.example.")
    if not config.TMDB_ENRICHED.exists():
        sys.exit("Run tmdb_enrichment.py first (it provides the imdb_id used to match films).")

    parsed = pd.read_pickle(config.PARSED)
    parsed, _ = apply_corrections(parsed, pd.read_csv(config.TMDB_ENRICHED))
    films = filter_modelable(parsed, verbose=False)      # only fetch films that can be modelled
    print(f"{len(films)} modelable films to score")

    omdb_cache = config.CACHE_DIR / "omdb"; omdb_cache.mkdir(parents=True, exist_ok=True)
    rt_cache = config.CACHE_DIR / "rt"; rt_cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session(); session.headers["User-Agent"] = USER_AGENT
    limiter = RateLimiter(args.omdb_rps)
    rt = RTScraper(rt_cache) if args.rt_fallback else None

    rows, new_calls, stopped = [], 0, False
    for i, f in enumerate(films.itertuples(index=False), 1):
        year = f.release_date.year
        row = {"id": f.id, "imdb_id": f.imdb_id if isinstance(f.imdb_id, str) else None,
               "rt_critic": None, "metacritic": None, "post_imdb_rating": None,
               "post_rt_audience": None, "year_match": None, "source": None}
        if row["imdb_id"] and not stopped:
            try:
                data, networked = fetch_omdb(session, limiter, omdb_key, row["imdb_id"], omdb_cache)
                new_calls += int(networked)
                parsed_scores = parse_omdb(data, year)
                row.update(parsed_scores)
                row["source"] = "omdb" if parsed_scores["rt_critic"] is not None or parsed_scores["metacritic"] is not None else None
            except DailyLimitReached:
                print("OMDb daily limit reached - saving progress. Re-run tomorrow (responses are cached).")
                stopped = True
            if args.limit and new_calls >= args.limit:
                print(f"--limit {args.limit} reached - saving progress. Re-run to continue.")
                stopped = True
        if rt is not None and row["rt_critic"] is None:
            res = rt.scrape(f.id, f.title, year)
            if res.get("rt_critic") is not None:
                row.update({"rt_critic": res["rt_critic"], "source": "rt_scrape"})
            if res.get("post_rt_audience") is not None:
                row["post_rt_audience"] = res["post_rt_audience"]
        rows.append(row)
        if i % 200 == 0:
            print(f"{i}/{len(films)} processed")

    out = pd.DataFrame(rows)
    out.to_csv(config.CRITIC_SCORES, index=False)
    compat = out[["id", "imdb_id", "rt_critic", "metacritic"]].merge(
        parsed[["id", "us_certification"]], on="id", how="left")
    compat.to_csv(config.ENRICHED_COMPAT, index=False)
    print(f"Saved {len(out)} rows -> {config.CRITIC_SCORES}")
    print(out[["rt_critic", "metacritic", "post_imdb_rating", "post_rt_audience"]].notna().mean().round(2))


if __name__ == "__main__":
    main()
