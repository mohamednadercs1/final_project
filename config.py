"""Central configuration shared by every stage of the pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("MOVIE_DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Files produced / consumed by the pipeline stages
RAW_MOVIES = DATA_DIR / "tmdb_5000_movies.csv"
RAW_CREDITS = DATA_DIR / "tmdb_5000_credits.csv"
PARSED = DATA_DIR / "movies_parsed.pkl"            # step 1 output
TMDB_ENRICHED = DATA_DIR / "tmdb_enriched.csv"     # step 2 output
CRITIC_SCORES = DATA_DIR / "critic_scores.csv"     # step 3 output
ENRICHED_COMPAT = DATA_DIR / "enriched.csv"        # step 3 output (read by the notebook)
MODELING_FRAME = DATA_DIR / "modeling_frame.pkl"   # step 4 output

# Data-quality thresholds (USD)
MIN_BUDGET = 100_000
MIN_REVENUE = 100_000

# Target definitions
REVENUE_THRESHOLD = 100_000_000   # success_rev: revenue >= $100M
ROI_THRESHOLD = 2.5               # success_roi: revenue >= 2.5 x budget
ROI_TIERS = (1.0, 2.5)            # tier: 0 Low (<1x), 1 Medium (1x-2.5x), 2 High (>=2.5x)
TIER_NAMES = ["Low", "Medium", "High"]

# Evaluation
TEST_FRACTION = 0.20              # most recent 20% of films form the temporal test set
CV_FOLDS = 5
SEED = 42

# Business costs in $ millions (editable; also exposed as CLI flags in train_models.py)
COST_FP = 25.0   # greenlit a film that flopped: average net loss
COST_FN = 10.0   # passed on a film that was a hit: forgone profit

# Anything measured AFTER release is excluded from every model (look-ahead bias)
POST_RELEASE_COLS = [
    "popularity", "vote_average", "vote_count",
    "post_vote_average", "post_vote_count", "post_popularity",
    "post_imdb_rating", "post_rt_audience", "revenue", "roi",
]

MAJOR_STUDIO_TOKENS = [
    "warner bros", "universal pictures", "paramount", "walt disney", "pixar", "marvel",
    "lucasfilm", "twentieth century fox", "20th century fox", "columbia pictures",
    "sony pictures", "tristar", "metro-goldwyn-mayer", "new line cinema", "dreamworks",
    "legendary", "lionsgate",
]


def load_env(path=None):
    """Load KEY=VALUE pairs from <project>/.env into os.environ (existing variables win)."""
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
