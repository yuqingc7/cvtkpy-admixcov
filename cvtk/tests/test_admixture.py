import numpy as np

from cvtk.admixture import decompose_admixture_covariance, infer_single_source_alphas, predict_admixture_frequencies


def test_prediction_and_alpha_direction():
    Q = np.array([[0.9, 0.1], [0.8, 0.2], [0.85, 0.15]])
    ref = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(predict_admixture_frequencies(Q, ref), [[0.1, 0.9], [0.2, 0.8], [0.15, 0.85]])
    alphas = infer_single_source_alphas(Q)
    np.testing.assert_allclose(alphas[0], [0.0, 1.0 / 9.0])
    np.testing.assert_allclose(alphas[1], [0.25, 0.0])


def test_decomposition_identity():
    af = np.array([[.1, .2, .7, .8], [.12, .22, .68, .78], [.15, .21, .66, .75]])
    Q = np.array([[.9, .1], [.8, .2], [.75, .25]])
    ref = np.array([[.1, .2, .8, .9], [.8, .7, .2, .1]])
    result = decompose_admixture_covariance(af, np.repeat(50., 3), Q, ref, np.full_like(ref, 100.))
    np.testing.assert_allclose(result["residual"], result["raw"] - result["admixture"] - result["drift_error"])
