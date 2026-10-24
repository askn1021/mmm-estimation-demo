"""Generate synthetic observations and fit a small Bayesian media mix model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp


@dataclass(frozen=True)
class Settings:
    seed: int = 2026
    weeks: int = 156
    train_weeks: int = 124
    grid_points: int = 51
    posterior_draws: int = 4000
    alpha: tuple[float, float] = (0.40, 0.65)
    beta: tuple[float, float] = (18.0, 26.0)
    saturation: tuple[float, float] = (60.0, 90.0)
    intercept: float = 30.0
    noise_sd: float = 2.5


def adstock(spend: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    spend = np.asarray(spend, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if spend.ndim != 2 or alpha.shape != (spend.shape[1],):
        raise ValueError("Spend must be a matrix with one alpha per channel.")
    if not np.isfinite(spend).all() or (spend < 0).any():
        raise ValueError("Spend must be finite and nonnegative.")
    if not np.isfinite(alpha).all() or ((alpha < 0) | (alpha >= 1)).any():
        raise ValueError("Alpha must be finite and in [0, 1).")
    state = np.zeros(spend.shape[1])
    values = np.empty_like(spend)
    for week, amount in enumerate(spend):
        state = amount + alpha * state
        values[week] = state
    return values


def design(spend: np.ndarray, alpha: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    carried = adstock(spend, alpha)
    response = carried / (carried + np.asarray(saturation))
    return np.column_stack((np.ones(len(spend)), response))


def generate_data(settings: Settings) -> pd.DataFrame:
    rng = np.random.default_rng(settings.seed)
    scales = np.array([0.45, 0.55])
    spend = np.array([35.0, 50.0]) * np.exp(
        rng.normal(size=(settings.weeks, 2)) * scales - scales**2 / 2
    )
    matrix = design(spend, np.array(settings.alpha), np.array(settings.saturation))
    mean = matrix @ np.array([settings.intercept, *settings.beta])
    return pd.DataFrame({
        "week": np.arange(1, settings.weeks + 1),
        "spend_a": spend[:, 0],
        "spend_b": spend[:, 1],
        "outcome": mean + rng.normal(0, settings.noise_sd, settings.weeks),
        "true_mean": mean,
        "split": np.where(np.arange(settings.weeks) < settings.train_weeks, "train", "test"),
    })


def fit(spend: np.ndarray, outcome: np.ndarray, settings: Settings) -> dict:
    spend = np.asarray(spend, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if outcome.shape != (len(spend),) or not np.isfinite(outcome).all():
        raise ValueError("Outcome must be a finite vector matching spend.")
    if settings.grid_points < 3 or settings.noise_sd <= 0:
        raise ValueError("Use at least three grid points and positive noise SD.")
    axis = (np.arange(settings.grid_points) + 0.5) / settings.grid_points
    pairs = np.array(np.meshgrid(axis, axis, indexing="ij")).reshape(2, -1).T
    prior_precision = np.eye(3) / 50.0**2
    means, covariances, scores = [], [], []
    noise_variance = settings.noise_sd**2
    for alpha in pairs:
        matrix = design(spend, alpha, np.array(settings.saturation))
        precision = prior_precision + matrix.T @ matrix / noise_variance
        covariance = np.linalg.inv(precision)
        natural_mean = matrix.T @ outcome / noise_variance
        mean = covariance @ natural_mean
        score = 0.5 * natural_mean @ mean - 0.5 * np.linalg.slogdet(precision)[1]
        means.append(mean)
        covariances.append(covariance)
        scores.append(score)
    weights = np.exp(np.array(scores) - logsumexp(scores))
    return {
        "alpha": pairs,
        "weights": weights,
        "means": np.array(means),
        "covariances": np.array(covariances),
    }


def draw_parameters(posterior: dict, settings: Settings) -> np.ndarray:
    rng = np.random.default_rng(settings.seed + 1)
    indices = rng.choice(len(posterior["weights"]), settings.posterior_draws,
                         p=posterior["weights"])
    normals = rng.normal(size=(settings.posterior_draws, 3))
    factors = np.linalg.cholesky(posterior["covariances"][indices])
    coefficients = posterior["means"][indices] + np.einsum("nij,nj->ni", factors, normals)
    return np.column_stack((posterior["alpha"][indices], coefficients))


def predict(spend: np.ndarray, posterior: dict, draws: np.ndarray,
            settings: Settings) -> pd.DataFrame:
    mean = np.zeros(len(spend))
    for index, alpha in enumerate(posterior["alpha"]):
        matrix = design(spend, alpha, np.array(settings.saturation))
        mean += posterior["weights"][index] * (matrix @ posterior["means"][index])
    latent = np.empty((len(draws), len(spend)))
    for index, row in enumerate(draws):
        latent[index] = design(spend, row[:2], np.array(settings.saturation)) @ row[2:]
    rng = np.random.default_rng(settings.seed + 2)
    predictive = latent + rng.normal(0, settings.noise_sd, latent.shape)
    low, high = np.quantile(predictive, [0.05, 0.95], axis=0)
    return pd.DataFrame({"prediction": mean, "predictive_low": low, "predictive_high": high})


def parameter_summary(draws: np.ndarray, settings: Settings) -> pd.DataFrame:
    names = ["alpha_a", "alpha_b", "intercept", "beta_a", "beta_b"]
    truth = [*settings.alpha, settings.intercept, *settings.beta]
    low, high = np.quantile(draws, [0.05, 0.95], axis=0)
    return pd.DataFrame({
        "parameter": names, "truth": truth, "posterior_mean": draws.mean(axis=0),
        "credible_low": low, "credible_high": high,
    })


def evaluate(data: pd.DataFrame, predictions: pd.DataFrame,
             settings: Settings) -> dict:
    test = data["split"].eq("test").to_numpy()
    actual = data.loc[test, "outcome"].to_numpy()
    estimated = predictions.loc[test, "prediction"].to_numpy()
    residual = actual - estimated
    baseline = float(data.loc[~test, "outcome"].mean())
    covered = (actual >= predictions.loc[test, "predictive_low"].to_numpy()) & (
        actual <= predictions.loc[test, "predictive_high"].to_numpy()
    )
    return {
        "train_weeks": int((~test).sum()), "test_weeks": int(test.sum()),
        "test_rmse": float(np.sqrt(np.mean(residual**2))),
        "test_mae": float(np.mean(np.abs(residual))),
        "train_mean_baseline_rmse": float(np.sqrt(np.mean((actual - baseline)**2))),
        "test_predictive_90_coverage": float(covered.mean()),
        "test_predictive_90_mean_width": float(np.mean(
            predictions.loc[test, "predictive_high"] - predictions.loc[test, "predictive_low"]
        )),
    }
