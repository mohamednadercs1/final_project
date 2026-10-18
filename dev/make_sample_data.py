"""Generate SYNTHETIC data in the exact Kaggle TMDB-5000 format, for smoke-testing without any
download or API key. The numbers are fake - never report results from this data.

    python dev/make_sample_data.py --n 2500            # writes data/tmdb_5000_*.csv
    python dev/make_sample_data.py --n 2500 --extras   # + tmdb_enriched.csv and critic_scores.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family",
          "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller", "War"]
GENRE_EFFECT = {"Animation": .35, "Family": .3, "Adventure": .25, "Horror": .3, "Documentary": -.4, "Drama": -.2}
MAJORS = ["Warner Bros.", "Universal Pictures", "Paramount Pictures", "Walt Disney Pictures",
          "Columbia Pictures", "Twentieth Century Fox Film Corporation", "New Line Cinema"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--extras", action="store_true", help="also write tmdb_enriched.csv + critic_scores.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    directors = [f"Director {i}" for i in range(350)]
    skill = rng.normal(0, .5, len(directors))
    actors = [f"Actor {i}" for i in range(900)]
    fame = rng.normal(0, .4, len(actors))
    indies = [f"Indie Studio {i} (Pictures)" for i in range(60)]

    movies, credits, enriched, critics = [], [], [], []
    for i in range(args.n):
        mid = 1000 + i
        date = pd.Timestamp("1995-01-01") + pd.Timedelta(days=int(rng.integers(0, 22 * 365)))
        gs = [str(g) for g in rng.choice(GENRES, size=rng.integers(1, 4), replace=False)]
        budget = float(np.clip(rng.lognormal(np.log(25e6), 1.0), 2e5, 3e8))
        sequel = rng.random() < .12
        d = int(rng.integers(len(directors)))
        a = rng.choice(len(actors), 3, replace=False)
        studios = ([str(rng.choice(MAJORS))] if rng.random() < .45 else []) + \
                  [str(s) for s in rng.choice(indies, size=rng.integers(1, 3))]
        summer = date.month in (5, 6, 7)
        latent = (skill[d] + .3 * fame[a].mean() + .5 * sequel + .35 * (studios[0] in MAJORS)
                  + .3 * summer + sum(GENRE_EFFECT.get(g, 0) for g in gs) / len(gs))
        revenue = budget * float(np.exp(rng.normal(.45 + latent, .9)))
        title = f"Movie {i}" + (" 2" if sequel else "")
        lang = "en" if rng.random() < .85 else "fr"

        zero_b, zero_r = rng.random() < .18, rng.random() < .12
        movies.append({
            "id": mid, "title": title, "original_language": lang, "status": "Released",
            "release_date": date.strftime("%Y-%m-%d"), "runtime": int(rng.normal(108, 15)),
            "budget": 0 if zero_b else int(budget), "revenue": 0 if zero_r else int(revenue),
            "popularity": float(rng.gamma(2, 10)), "vote_count": int(rng.integers(5, 9000)),
            "vote_average": float(np.clip(6 + latent + rng.normal(0, .6), 2, 9.5)),
            "genres": json.dumps([{"id": GENRES.index(g), "name": g} for g in gs]),
            "keywords": json.dumps([{"id": 1, "name": "sequel" if sequel else "independent film"}]),
            "production_companies": json.dumps([{"name": s, "id": k} for k, s in enumerate(studios)]),
        })
        credits.append({
            "movie_id": mid, "title": title,
            "cast": json.dumps([{"name": actors[x], "order": o, "id": int(x)} for o, x in enumerate(a)]),
            "crew": json.dumps([{"job": "Director", "name": directors[d], "department": "Directing"}]),
        })
        if args.extras:
            enriched.append({"id": mid, "imdb_id": f"tt{9000000 + i}",
                             "tmdb_budget": int(budget) if (zero_b and rng.random() < .5) else np.nan,
                             "tmdb_revenue": np.nan, "tmdb_runtime": np.nan,
                             "tmdb_release_date": date.strftime("%Y-%m-%d"),
                             "us_certification": rng.choice(["G", "PG", "PG-13", "R", None], p=[.04, .2, .4, .3, .06])})
            has = rng.random() < .75
            critics.append({"id": mid, "imdb_id": f"tt{9000000 + i}",
                            "rt_critic": float(np.clip(55 + 20 * latent + rng.normal(0, 14), 0, 100)) if has else np.nan,
                            "metacritic": float(np.clip(55 + 15 * latent + rng.normal(0, 12), 0, 100))
                            if has and rng.random() < .8 else np.nan,
                            "post_imdb_rating": 6.0, "post_rt_audience": 70.0, "year_match": True, "source": "omdb"})

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(movies).to_csv(config.RAW_MOVIES, index=False)
    pd.DataFrame(credits).to_csv(config.RAW_CREDITS, index=False)
    if args.extras:
        pd.DataFrame(enriched).to_csv(config.TMDB_ENRICHED, index=False)
        pd.DataFrame(critics).to_csv(config.CRITIC_SCORES, index=False)
    print(f"Wrote {args.n} synthetic films to {config.DATA_DIR}")


if __name__ == "__main__":
    main()
