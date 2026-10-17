"""Covariance decomposition for temporal samples with changing ancestry."""

import numpy as np

from cvtk.cov import temporal_replicate_cov


def predict_admixture_frequencies(Q, reference_frequencies):
    Q = np.asarray(Q, dtype=float)
    ref = np.asarray(reference_frequencies, dtype=float)
    if Q.ndim != 2 or ref.ndim != 2 or Q.shape[1] != ref.shape[0]:
        raise ValueError("Q and reference_frequencies have incompatible shapes")
    if not np.allclose(Q.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each Q row must sum to one")
    return Q @ ref


def raw_temporal_covariance(frequencies, diploids, bias_correction=True):
    frequencies = np.asarray(frequencies, dtype=float)
    diploids = np.asarray(diploids, dtype=float)
    if frequencies.ndim != 2 or diploids.shape != (frequencies.shape[0],):
        raise ValueError("frequencies must be T x L and diploids must have length T")
    cov = temporal_replicate_cov(
        frequencies[np.newaxis, ...],
        diploids=diploids[np.newaxis, :, np.newaxis],
        standardize=False,
        bias_correction=bias_correction,
        use_masked=True,
    )
    return np.atleast_2d(cov)


def admixture_covariance(Q, reference_frequencies, reference_allele_counts=None,
                         bias_correction=True):
    Q = np.asarray(Q, dtype=float)
    ref = np.asarray(reference_frequencies, dtype=float)
    predicted = predict_admixture_frequencies(Q, ref)
    cov = np.atleast_2d(
        np.ma.cov(np.ma.masked_invalid(np.diff(predicted, axis=0)), bias=True).data
    )
    if bias_correction:
        counts = np.asarray(reference_allele_counts, dtype=float)
        if counts.shape != ref.shape:
            raise ValueError("reference_allele_counts must match reference_frequencies")
        denominator = counts - 1.0
        denominator[denominator <= 0] = np.nan
        source_bias = np.nanmean(ref * (1.0 - ref) / denominator, axis=1)
        dQ = np.diff(Q, axis=0)
        cov -= (dQ * source_bias[np.newaxis, :]) @ dQ.T
    return cov


def infer_single_source_alphas(Q, tolerance=1e-10):
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or not np.allclose(Q.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Q must be two-dimensional and each row must sum to one")
    alphas = np.zeros((Q.shape[0] - 1, Q.shape[1]))
    for interval, changes in enumerate(np.diff(Q, axis=0)):
        incoming = np.flatnonzero(changes > tolerance)
        if incoming.size > 1:
            raise ValueError("more than one source increases in interval {}".format(interval + 1))
        if incoming.size:
            source = incoming[0]
            denominator = 1.0 - Q[interval, source]
            if denominator <= tolerance:
                raise ValueError("incoming source has no available ancestry fraction")
            alphas[interval, source] = changes[source] / denominator
    return alphas


def solve_drift_variances(diagonal_variance, alphas):
    diagonal_variance = np.asarray(diagonal_variance, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    size = diagonal_variance.size
    if alphas.shape[0] != size:
        raise ValueError("alphas must have one row per interval")
    total = alphas.sum(axis=1)
    design = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1):
            if j == i:
                design[i, j] = 1.0
            elif j == i - 1:
                design[i, j] = total[i] ** 2
            else:
                design[i, j] = total[i] ** 2 * np.prod((1.0 - total[j + 1:i]) ** 2)
    return np.linalg.solve(design, diagonal_variance)


def drift_error_matrix(drift_variances, alphas):
    drift_variances = np.asarray(drift_variances, dtype=float)
    total = np.asarray(alphas, dtype=float).sum(axis=1)
    size = drift_variances.size
    out = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            value = 0.0
            for k in range(i + 1):
                mi = np.prod(1.0 - total[k + 1:i]) if i - k > 1 else 1.0
                mj = np.prod(1.0 - total[k + 1:j]) if j - k > 1 else 1.0
                value += drift_variances[k] * ((-total[i]) if i > k else 1.0) * mi * (-total[j]) * mj
            out[i, j] = out[j, i] = value
    return out


def decompose_admixture_covariance(frequencies, diploids, Q, reference_frequencies,
                                    reference_allele_counts, bias_correction=True,
                                    alphas=None):
    raw = raw_temporal_covariance(frequencies, diploids, bias_correction)
    admix = admixture_covariance(
        Q, reference_frequencies, reference_allele_counts, bias_correction
    )
    if raw.shape != admix.shape:
        raise ValueError("raw and admixture covariance shapes differ")
    if alphas is None:
        alphas = infer_single_source_alphas(Q)
    drift_variances = solve_drift_variances(np.diag(raw - admix), alphas)
    drift_error = drift_error_matrix(drift_variances, alphas)
    return {
        "raw": raw,
        "admixture": admix,
        "drift_error": drift_error,
        "residual": raw - admix - drift_error,
        "drift_variances": drift_variances,
        "alphas": alphas,
    }


def matrix_sum(matrix, include_diagonal=False):
    matrix = np.asarray(matrix)
    return np.nansum(matrix) if include_diagonal else np.nansum(matrix - np.diag(np.diag(matrix)))


def covariance_statistics(raw, admixture, drift_error):
    residual = raw - admixture - drift_error
    total = np.nansum(raw)
    return {
        "total_variance": total,
        "G_nc": matrix_sum(raw) / total,
        "G_nde": matrix_sum(raw - admixture) / total,
        "G": matrix_sum(residual) / total,
        "Ap": matrix_sum(admixture, include_diagonal=True) / total,
    }
