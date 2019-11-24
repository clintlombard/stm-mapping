import logging

from enum import Enum, unique

import numpy as np

from scipy.special import digamma, loggamma

logger = logging.getLogger(__name__)


@unique
class DistanceMetric(Enum):
    BHATTACHARYYA = 0
    MAHALANOBIS = 1
    SYMMETRIC_KLD = 2
    EXCLUSIVE_KLD = 3
    INCLUSIVE_KLD = 4


def distance_gauss(
    p_m: np.ndarray,
    p_C: np.ndarray,
    q_m: np.ndarray,
    q_C: np.ndarray,
    metric: DistanceMetric = DistanceMetric.EXCLUSIVE_KLD,
) -> float:
    """Calculate D(p||q) between two multivariate Gaussian distributions.

    This assumes the Gaussians have the same variable ordering and scope.

    """
    dim = p_m.size
    assert p_m.shape == (dim, 1)
    assert q_m.shape == (dim, 1)
    assert p_C.shape == (dim, dim)
    assert q_C.shape == (dim, dim)

    if metric == DistanceMetric.BHATTACHARYYA:
        C = (p_C + q_C) / 2
        d = 0.125 * (p_m - q_m).T.dot(np.linalg.inv(C)).dot(p_m - q_m)
        d += 0.5 * np.log(np.linalg.det(C))
        d += -0.25 * np.log(np.linalg.det(p_C))
        d += -0.25 * np.log(np.linalg.det(q_C))
    elif metric == DistanceMetric.MAHALANOBIS:
        d = np.sqrt((p_m - q_m).T.dot(np.linalg.inv(p_C)).dot(p_m - q_m))
        d += np.sqrt((p_m - q_m).T.dot(np.linalg.inv(q_C)).dot(p_m - q_m))
        d /= p_m.size ** 0.5
    elif metric == DistanceMetric.SYMMETRIC_KLD:
        p_K = np.linalg.inv(p_C)
        q_K = np.linalg.inv(q_C)
        d = 0.5 * (p_m - q_m).T.dot(p_K).dot(p_m - q_m)
        d += 0.5 * (p_m - q_m).T.dot(q_K).dot(p_m - q_m)
        d += 0.5 * np.trace(p_K.dot(q_C))
        d += 0.5 * np.trace(q_K.dot(p_C))
        d -= dim
    elif metric == DistanceMetric.EXCLUSIVE_KLD:
        q_K = np.linalg.inv(q_C)
        p_C_det = np.linalg.det(p_C)
        q_C_det = np.linalg.det(q_C)
        d = 0.5 * np.trace(q_K.dot(p_C))
        d += 0.5 * (p_m - q_m).T.dot(q_K).dot(p_m - q_m)
        d += 0.5 * (np.log(q_C_det) - np.log(p_C_det))
        d -= dim / 2
    elif metric == DistanceMetric.INCLUSIVE_KLD:
        p_K = np.linalg.inv(p_C)
        q_C_det = np.linalg.det(q_C)
        p_C_det = np.linalg.det(p_C)
        d = 0.5 * np.trace(p_K.dot(q_C))
        d += 0.5 * (q_m - p_m).T.dot(p_K).dot(q_m - p_m)
        d += 0.5 * (np.log(q_C_det) - np.log(q_C_det))
        d -= dim / 2
    else:
        raise ValueError("Invalid distance metric")

    return d.item()


def distance_invgamma(
    p_a: float,
    p_b: float,
    q_a: float,
    q_b: float,
    metric: DistanceMetric = DistanceMetric.EXCLUSIVE_KLD,
) -> float:
    """Calculate D(p||q) between two inverse-gamma distributions."""

    if metric == DistanceMetric.BHATTACHARYYA:
        raise NotImplementedError()
    elif metric == DistanceMetric.MAHALANOBIS:
        raise NotImplementedError()
    elif metric == DistanceMetric.SYMMETRIC_KLD:
        d = (p_a - q_a) * digamma(p_a)
        d += (q_a - p_a) * digamma(q_a)
        d += q_a * (np.log(p_b) - np.log(q_b))
        d += p_a * (np.log(q_b) - np.log(p_b))
        d += p_a * (q_b - p_b) / p_b
        d += q_a * (p_b - q_b) / q_b
    elif metric == DistanceMetric.EXCLUSIVE_KLD:
        d = (p_a - q_a) * digamma(p_a)
        d -= loggamma(p_a)
        d += loggamma(q_a)
        d += q_a * (np.log(p_b) - np.log(q_b))
        d += p_a * (q_b - p_b) / p_b
    elif metric == DistanceMetric.INCLUSIVE_KLD:
        d = (q_a - p_a) * digamma(q_a)
        d -= loggamma(q_a)
        d += loggamma(p_a)
        d += p_a * (np.log(q_b) - np.log(p_b))
        d += q_a * (p_b - q_b) / q_b
    else:
        raise ValueError("Invalid distance metric")

    try:
        return d.item()
    except:
        return d


def compare_maps(submap_0, submap_1, metric: DistanceMetric = DistanceMetric.EXCLUSIVE_KLD):
    """Calculate the probabilistic distance between two maps."""

    distances = {}
    for surfel_id in submap_0.keys():
        surfel_0 = submap_0[surfel_id]
        surfel_1 = submap_1[surfel_id]
        m0 = surfel_0.bel_h_m
        C0 = surfel_0.bel_h_C
        m1 = surfel_1.bel_h_m
        C1 = surfel_1.bel_h_C

        a0 = surfel_0.bel_v_a
        b0 = surfel_0.bel_v_b
        a1 = surfel_1.bel_v_a
        b1 = surfel_1.bel_v_b

        d = distance_gauss(m0, C0, m1, C1, metric)
        d += distance_invgamma(a0, b0, a1, b1, metric)

        distances[surfel_id] = d

    return distances
