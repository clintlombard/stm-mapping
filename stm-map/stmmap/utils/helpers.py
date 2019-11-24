import numpy as np

from scipy import linalg


def cov_corrcoef(C):
    stds = np.sqrt(np.diag(C))
    corr_coeffs = np.ones_like(C)
    N = C.shape[0]
    for i in range(N):
        for j in range(N):
            corr_coeffs[i, j] = C[i, j] / (stds[i] * stds[j])
    return corr_coeffs


def stable_inverse(C):
    L = np.linalg.cholesky(C)
    L_inv = np.linalg.inv(L)
    if len(C.shape) > 2:
        K = np.einsum("ikj,ikm->ijm", L_inv, L_inv)
    else:
        K = L_inv.T.dot(L_inv)
    return K


def covToCan(m, C):
    K = np.linalg.inv(C)
    h = K.dot(m)
    return h, K


def canToCov(h, K):
    C = np.linalg.inv(K)
    m = C.dot(h)
    return m, C


def calcSigCan(hx, Kxx):
    m, c = canToCov(hx, Kxx)
    return calcSigCov(m, c)


def multiplyCan(h1, K1, h2, K2):
    h3 = h1 + h2
    K3 = K1 + K2
    return h3, K3


def multiplyCov(m1, C1, m2, C2):
    h1, K1 = covToCan(m1, C1)
    h2, K2 = covToCan(m2, C2)
    h3 = h1 + h2
    K3 = K1 + K2
    m3, C3 = canToCov(h3, K3)
    return m3, C3


def divideCan(h1, K1, h2, K2):
    h3 = h1 - h2
    K3 = K1 - K2
    return h3, K3


def divideCov(m1, C1, m2, C2):
    h1, K1 = covToCan(m1, C1)
    h2, K2 = covToCan(m2, C2)
    h3 = h1 - h2
    K3 = K1 - K2
    m3, C3 = canToCov(h3, K3)
    return m3, C3


def divideCovCan(m1, C1, m2, C2):
    h1, K1 = covToCan(m1, C1)
    h2, K2 = covToCan(m2, C2)
    h3 = h1 - h2
    K3 = K1 - K2
    return h3, K3
