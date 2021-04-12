import pandas, math, os
from pyentrp import entropy
import numpy as np
from scipy.special import gamma, psi
from numpy import pi
from sklearn.neighbors import NearestNeighbors
from config import FEATURE_PATH
from math import log, floor

def kraskov_entropy(d1):
    k = 4
    def nearest_distances(X, k):
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        d, _ = knn.kneighbors(X)
        return d[:, -1]
    def entropy(X, k):
        r = nearest_distances(X, k)
        n, d = X.shape
        volume_unit_ball = (pi ** (.5 * d)) / gamma(.5 * d + 1)
        return (d * np.mean(np.log(r + np.finfo(X.dtype).eps)) + np.log(volume_unit_ball) + psi(n) - psi(k))
    kd1 = []
    for i in range(d1.shape[0]):
        x = d1[i]
        x = np.array(x).reshape(-1, 1)
        kd1.append(entropy(x, k))
    return (kd1)

def renyi_entropy(d1):
    d1 = np.rint(d1)
    rend1 = []
    alpha = 2
    for i in range(d1.shape[0]):
        X = d1[i]
        data_set = list(set(X))
        freq_list = []
        for entry in data_set:
            counter = 0.
            for i in X:
                if i == entry:
                    counter += 1
            freq_list.append(float(counter) / len(X))
        summation = 0
        for freq in freq_list:
            summation += math.pow(freq, alpha)
        Renyi_En = (1 / float(1 - alpha)) * (math.log(summation, 2))
        rend1.append(Renyi_En)
    return (rend1)

def permu(d1):
    pd1 = []
    for i in range(d1.shape[0]):
        X = d1[i]
        pd1.append(entropy.permutation_entropy(X, 3, 1))
    return (pd1)

def sampl(d1):
    sa1 = []
    for i in range(d1.shape[0]):
        X = d1[i]
        std_X = np.std(X)
        ee = entropy.sample_entropy(X, 2, 0.2 * std_X)
        sa1.append(ee[0])
    return (sa1)

def shan(d1):
    sh1 = []
    d1 = np.rint(d1)
    for i in range(d1.shape[0]):
        X = d1[i]
        sh1.append(entropy.shannon_entropy(X))
    return (sh1)

def energy_ext(d1):
    pd1 = []
    for i in range(d1.shape[0]):
        X = np.array(d1[i])
        eng = np.sum(X ** 2)
        pd1.append(eng)
    return (pd1)

def _embed(x, order=3, delay=1):
    """Time-delay embedding.
    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.
    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

def svd_entropy(d1, order=3, delay=1, normalize=False):
    svd1 =[]
    for i in range(d1.shape[0]):
        x = np.array(d1[i])
        mat = _embed(x,order=order, delay=delay)
        W = np.linalg.svd(mat, compute_uv=False)
        W /= sum(W)
        svd_e = -np.multiply(W, np.log2(W)).sum()
        if normalize:
            svd_e /= np.log2(order)
        svd1.append(svd_e)
    return (svd1)

def petrosian_fd(d1):
    pfd1 = []
    for i in range(d1.shape[0]):
        n = len(d1[i])
        diff = np.ediff1d(d1[i])
        N_delta = (diff[1:-1] * diff[0:-2] < 0).sum()
        pfd = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))
        pfd1.append(pfd)
    return (pfd1)

def katz_fd(d1):
    kfd1 = []
    for i in range(d1.shape[0]):
        x = np.array(d1[i])
        dists = np.abs(np.ediff1d(x))
        ll = dists.sum()
        ln = np.log10(np.divide(ll, dists.mean()))
        aux_d = x - x[0]
        d = np.max(np.abs(aux_d[1:]))
        kfd = np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))
        kfd1.append(kfd)
    return (kfd1)

def _linear_regression(x, y):
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept

def _higuchi_fd(x, kmax):
    """Utility function for `higuchi_fd`.
    """
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi

def higuchi_fd(d1, kmax=10):
    hfd1 = []
    for i in range(d1.shape[0]):
        x = np.asarray(d1[i], dtype=np.float64)
        kmax = int(kmax)
        hfd = _higuchi_fd(x, kmax)
        hfd1.append(hfd)
    return (hfd1)

def extration(data):
    kra1 = kraskov_entropy(data)
    ren1 = renyi_entropy(data)
    per1 = permu(data)
    sam1 = sampl(data)
    sha1 = shan(data)
    eng1 = energy_ext(data)
    svd1= svd_entropy(data, order=2)
    pfd1 = petrosian_fd(data)
    kfd1 = katz_fd(data)
    hfd1 = higuchi_fd(data)
    return (kra1, ren1, per1,sam1, sha1, eng1,svd1, pfd1, kfd1, hfd1)

def extraction_entropy_feature(x, fs):
    lowcut = 0.5
    highcut = 60
    db = pywt.Wavelet('db10')
    a4 = []
    d4 = []
    d3 = []
    d2 = []
    d1 = []
    x = butter_bandpass_filter(x, lowcut, highcut, fs, order=4)
    cA, cD4, cD3, cD2, cD1 = pywt.wavedec(x, db, level=4)
    a4.append(cA)
    d4.append(cD4)
    d3.append(cD3)
    d2.append(cD2)
    d1.append(cD1)
    efa4 = extration(np.array(a4))
    efd4 = extration(np.array(d4))
    efd3 = extration(np.array(d3))
    efd2 = extration(np.array(d2))
    efd1 = extration(np.array(d1))
    return (efa4, efd4, efd3, efd2, efd1)
print("Alldata Entropy Feature Finished successfully.")