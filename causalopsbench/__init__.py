"""CausalOpsBench benchmark scaffold."""

from causalopsbench.baselines import get_baseline
from causalopsbench.evaluate import evaluate_baseline, evaluate_predictions
from causalopsbench.generator import EpisodeGenerator
from causalopsbench.metrics import score_episode
from causalopsbench.schemas import Episode, Prediction, ScoreBreakdown

__all__ = [
    "Episode",
    "EpisodeGenerator",
    "Prediction",
    "ScoreBreakdown",
    "evaluate_baseline",
    "evaluate_predictions",
    "get_baseline",
    "score_episode",
]

__version__ = "0.1.0"
