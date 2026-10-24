from dataclasses import replace

import numpy as np
import pytest

from scipy.special import softmax
from scipy.stats import multivariate_normal

from mmm_demo.analysis import Settings, adstock, design, draw_parameters, fit, generate_data, predict


def test_adstock_impulse():
    actual = adstock(np.array([[10.0], [0.0], [0.0]]), np.array([0.5]))
    np.testing.assert_allclose(actual[:, 0], [10.0, 5.0, 2.5])


def test_adstock_does_not_use_future_spend():
    spend = np.ones((8, 2))
    changed = spend.copy()
    changed[5:] *= 100
    np.testing.assert_array_equal(adstock(spend, np.array([0.3, 0.6]))[:5],
                                  adstock(changed, np.array([0.3, 0.6]))[:5])


def test_invalid_spend_rejected():
    with pytest.raises(ValueError):
        adstock(np.array([[-1.0, 2.0]]), np.array([0.3, 0.6]))


def test_posterior_and_prediction():
    settings = replace(Settings(), weeks=40, train_weeks=30, grid_points=9, posterior_draws=200)
    data = generate_data(settings)
    spend = data[["spend_a", "spend_b"]].to_numpy()
    posterior = fit(spend[:30], data.outcome.to_numpy()[:30], settings)
    assert np.isclose(posterior["weights"].sum(), 1)
    assert np.isfinite(posterior["weights"]).all()
    draws = draw_parameters(posterior, settings)
    prediction = predict(spend, posterior, draws, settings)
    assert np.isfinite(prediction.to_numpy()).all()
    assert (prediction.predictive_low < prediction.predictive_high).all()
    assert draws.shape == (200, 5)


def test_inference_does_not_use_simulation_truth():
    settings = replace(Settings(), weeks=24, train_weeks=18, grid_points=7)
    data = generate_data(settings)
    spend = data[["spend_a", "spend_b"]].to_numpy()[:18]
    outcome = data.outcome.to_numpy()[:18]
    original = fit(spend, outcome, settings)
    changed = fit(spend, outcome, replace(settings, alpha=(0.1, 0.2), beta=(1, 2), intercept=0))
    np.testing.assert_array_equal(original["weights"], changed["weights"])
    np.testing.assert_array_equal(original["means"], changed["means"])


def test_integrated_likelihood_matches_direct_gaussian():
    settings = replace(Settings(), weeks=12, train_weeks=10, grid_points=5)
    data = generate_data(settings)
    spend = data[["spend_a", "spend_b"]].to_numpy()[:10]
    outcome = data.outcome.to_numpy()[:10]
    posterior = fit(spend, outcome, settings)
    log_likelihood = []
    for alpha in posterior["alpha"]:
        matrix = design(spend, alpha, np.array(settings.saturation))
        covariance = settings.noise_sd**2 * np.eye(10) + 50.0**2 * matrix @ matrix.T
        log_likelihood.append(multivariate_normal.logpdf(outcome, mean=np.zeros(10), cov=covariance))
    np.testing.assert_allclose(posterior["weights"], softmax(log_likelihood), rtol=1e-8, atol=1e-10)
