import numpy as np,glob
from scipy.signal import detrend
import nolds,math
from math import log, floor
import pandas as pd
import os,time
from config import DATA_ROOT_PATH,FEATURE_PATH

def mean(data):
    return np.mean(abs(data), axis=0)

def var(data):
    return np.var(data, axis=0)

def cv(data):
    return mean(data)/np.std(data)

def skewness(data):
    s = pd.Series(data)
    return s.skew()

def kurtosis(data):
    s = pd.Series(data)
    return s.kurt()

# 上四分位值Q3与下四分位值Q1之间的差称为四分位距（IQR）,即IQR=Q3-Q1.
def IQR(data):
    a, b = np.percentile(data, [75, 25])
    return a - b

def hjorth(data):
    '''
    :param input: [batch signal]
    :return:
    '''

    lenth = len(data)
    diff_input = np.diff(data)
    diff_diffinput = np.diff(diff_input)

    hjorth_activity = np.var(data)
    hjorth_mobility = np.sqrt(np.var(diff_input) / hjorth_activity)
    hjorth_diffmobility = np.sqrt(np.var(diff_diffinput) / np.var(diff_input))
    hjorth_complexity = hjorth_diffmobility / hjorth_mobility

    return hjorth_activity, hjorth_mobility, hjorth_complexity

def ZeroCR(data,frameSize = 256,overLap = 0):
    wlen = len(data)
    step = frameSize - overLap
    frameNum = math.ceil(wlen/step)
    zcr = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = data[np.arange(i*step,min(i*step+frameSize,wlen))]
        curFrame = curFrame - np.mean(curFrame) # zero-justified
        zcr[i] = sum(curFrame[0:-1]*curFrame[1::]<=0)
    return np.mean(zcr)

def hurst(data):
    return nolds.hurst_rs(data)

def dfa(data):
    return nolds.dfa(data)

def petrosian_fd(data):
    n = len(data)
    diff = np.ediff1d(data)
    N_delta = (diff[1:-1] * diff[0:-2] < 0).sum()
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))

def katz_fd(data):
    x = np.array(data)
    dists = np.abs(np.ediff1d(x))
    ll = dists.sum()
    ln = np.log10(np.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = np.max(np.abs(aux_d[1:]))
    return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))

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

def higuchi_fd(x, kmax=10):
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    return _higuchi_fd(x, kmax)

def extraction_time_feature(data):
    mean1 = mean(data)
    var1 = var(data)
    cv1 = cv(data)
    skew1 = skewness(data)
    kurt1 = kurtosis(data)
    IQR1 = IQR(data)
    h_a,h_m,h_c = hjorth(data)
    ZCR1 = ZeroCR(data)
    hurst1 = hurst(data)
    dfa1 = dfa(data)
    pfd1 = petrosian_fd(data)
    kfd1 = katz_fd(data)
    hfd1 = higuchi_fd(data)
    return mean1,var1,cv1,skew1,kurt1,IQR1,h_a,h_m,h_c,ZCR1,hurst1,dfa1,pfd1,kfd1,hfd1

print("Here is Time domain feature extration.")