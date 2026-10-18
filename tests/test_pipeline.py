"""Fast unit tests (no network, no real data):  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_cleaning as dc          # noqa: E402
import features as ft               # noqa: E402
import scraper                      # noqa: E402
import tmdb_enrichment as te        # noqa: E402


def test_json_parsing_is_safe():
    assert dc.parse_json_list('[{"name": "Action"}]') == [{"name": "Action"}]
    assert dc.parse_json_list("[{'name': 'Drama'}]") == [{"name": "Drama"}]   # python-literal fallback
    assert dc.parse_json_list("garbage") == [] and dc.parse_json_list(np.nan) == []
    cast = '[{"name": "B", "order": 1}, {"name": "A", "order": 0}, {"name": "C", "order": 2}, {"name": "D", "order": 3}]'
    assert dc.extract_top_cast(cast, 3) == ["A", "B", "C"]
    assert dc.extract_director('[{"job": "Producer", "name": "P"}, {"job": "Director", "name": "D"}]') == "D"


def test_zero_budget_becomes_missing_and_is_filtered():
    raw = pd.DataFrame({"id": [1, 2], "title": ["a", "b"], "status": ["Released"] * 2,
                        "release_date": ["2010-01-01"] * 2, "budget": [0, 5_000_000], "revenue": [9e6, 0],
                        "runtime": [100, 100], "original_language": ["en"] * 2, "popularity": [1, 1],
                        "vote_average": [6, 6], "vote_count": [10, 10], "genres": ["[]"] * 2, "keywords": ["[]"] * 2,
                        "production_companies": ["[]"] * 2, "cast": ["[]"] * 2, "crew": ["[]"] * 2})
    parsed = dc.parse_dataframe(raw)
    assert parsed["budget"].isna().sum() == 1 and parsed["revenue"].isna().sum() == 1
    assert dc.filter_modelable(parsed, verbose=False).empty


def test_targets_and_tiers():
    df = pd.DataFrame({"budget": [10e6, 10e6, 100e6], "revenue": [5e6, 30e6, 150e6],
                       "vote_average": [5, 6, 7]})
    t = ft.add_targets(df)
    assert t["success_roi"].tolist() == [0, 1, 0]            # >= 2.5x budget
    assert t["success_rev"].tolist() == [0, 0, 1]            # >= $100M
    assert t["tier"].tolist() == [0, 2, 1]                   # <1x, >=2.5x, 1x-2.5x


def test_history_features_use_only_earlier_films():
    df = pd.DataFrame({
        "release_date": pd.to_datetime(["2000-01-01", "2001-01-01", "2002-01-01"]),
        "director": ["Dir", "Dir", "Dir"], "cast_top3": [["X"], ["X"], ["X"]],
        "studios_list": [["S"], ["S"], ["S"]],
        "log_roi": [1.0, 3.0, 5.0], "log_rev": [10.0, 12.0, 14.0], "vote": [6.0, 7.0, 8.0]})
    h = ft.add_history_features(df)
    assert np.isnan(h.loc[0, "director_prior_log_roi"]) and h.loc[0, "director_n_prior"] == 0
    assert h.loc[1, "director_prior_log_roi"] == 1.0         # only film 0
    assert h.loc[2, "director_prior_log_roi"] == 2.0         # mean of films 0 and 1, NOT its own 5.0
    assert h["cast_n_prior"].tolist() == [0, 1, 2]


def test_temporal_split_is_chronological():
    df = pd.DataFrame({"id": range(10), "release_date": pd.date_range("2000", periods=10, freq="365D")[::-1]})
    train, test = ft.temporal_split(df, 0.2)
    assert train["release_date"].max() <= test["release_date"].min() and len(test) == 2


def test_tmdb_extract_and_corrections():
    js = {"imdb_id": "tt1", "budget": 0, "revenue": 5_000_000, "runtime": 100, "release_date": "2010-05-01",
          "production_companies": [{"name": "A"}, {"name": "B"}],
          "release_dates": {"results": [{"iso_3166_1": "US", "release_dates": [{"certification": ""}, {"certification": "PG-13"}]}]},
          "vote_average": 7.1}
    fields = te.extract_fields(js)
    assert fields["us_certification"] == "PG-13" and fields["tmdb_n_companies"] == 2 and np.isnan(fields["tmdb_budget"])
    assert te.extract_fields({"_missing": True}) == {}
    base = pd.DataFrame({"id": [1], "budget": [np.nan], "revenue": [np.nan], "runtime": [np.nan],
                         "release_date": [pd.NaT]})
    fixed, report = te.apply_corrections(base, pd.DataFrame([{"id": 1, **fields}]))
    assert fixed.loc[0, "revenue"] == 5_000_000 and report["revenue"] == 1 and report["budget"] == 0


def test_omdb_parsing_and_year_guard():
    payload = {"Response": "True", "Year": "2010", "Metascore": "N/A", "imdbRating": "7.5",
               "Ratings": [{"Source": "Rotten Tomatoes", "Value": "87%"}, {"Source": "Metacritic", "Value": "72/100"}]}
    ok = scraper.parse_omdb(payload, 2010)
    assert ok["rt_critic"] == 87.0 and ok["metacritic"] == 72.0 and ok["post_imdb_rating"] == 7.5
    bad = scraper.parse_omdb(payload, 1990)                   # different film with the same title -> rejected
    assert bad["rt_critic"] is None and bad["year_match"] is False
    assert scraper.parse_omdb({"Response": "False"}, 2010)["rt_critic"] is None


def test_rt_html_parser_requires_title_match():
    html = ('<script type="application/ld+json">{"@type": "Movie", "name": "Inception", '
            '"aggregateRating": {"ratingValue": "87"}}</script>')
    assert scraper.parse_rt_html(html, "Inception")["rt_critic"] == 87.0
    assert scraper.parse_rt_html(html, "Some Other Movie") == {}      # avoids false joins
    assert scraper.parse_rt_html("<html></html>", "Inception") == {}


def test_feature_builder_has_no_post_release_columns_and_safe_names():
    rows = []
    for i in range(60):
        rows.append({"id": i, "release_date": pd.Timestamp("2005-01-01") + pd.Timedelta(days=30 * i),
                     "budget": 1e7 + i * 1e6, "revenue": 3e7, "runtime": 100.0, "original_language": "en",
                     "genres_list": ["Science Fiction", "Action"], "studios_list": ["Metro-Goldwyn-Mayer (MGM)"],
                     "is_sequel": 0, "director": "D", "cast_top3": ["A"], "vote_average": 6.5,
                     "popularity": 1.0, "vote_count": 5})
    df = ft.add_history_features(ft.add_targets(pd.DataFrame(rows)))
    fb = ft.FeatureBuilder(min_genre_films=5).fit(df)
    X = fb.transform(df)
    assert not [c for c in X if c.startswith("post_") or c in ("popularity", "vote_average", "vote_count", "revenue", "roi")]
    assert all(c.replace("_", "").isalnum() for c in X.columns)
