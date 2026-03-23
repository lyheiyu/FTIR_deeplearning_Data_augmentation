
# !/usr/bin/python

import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from utils import utils
import random
def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
    i = np.arange(0, m)
    E = eye(m, format='csc')
    D = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * D.T * D))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    '''
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): print
            'WARING max iteration reached!'
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z


if __name__ == '__main__':
    '''
    Example usage and testing
    '''
    print
    'Testing...'
    from scipy.stats import norm
    import matplotlib.pyplot as plt

    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData2('D4_4_publication5.csv', 2, 1763)
    x = np.arange(0, 1000, 1)





    g1 = norm(loc=100, scale=1.0)  # generate three gaussian as a signal
    g2 = norm(loc=300, scale=3.0)
    g3 = norm(loc=750, scale=5.0)
    waveLength=np.array(waveLength,dtype=np.float)
    signal = g1.pdf(x) + g2.pdf(x) + g3.pdf(x)
    baseline1 = 5e-4 * waveLength + 0.2  # linear baseline
    baseline2 = 0.5 * np.sin(np.pi * waveLength / waveLength.max())  # sinusoidal baseline
    noise1 = np.random.random(waveLength.shape[0]) / 500
    noise = np.random.random(waveLength.shape[0]) / 500
    print
    'Generating simulated experiment'
    y1 = intensity[0] + baseline1 + noise1
    y2 = intensity[0] + baseline2 + noise1
    print
    from PLS import airPLS
    y1=np.array(y1,dtype=np.float)
    # intensityBaseline=[]
    # for item in y1:
    #     item = item - airPLS(item)
    #     intensityBaseline.append(item)
    'Removing baselines'
    c1 = y1 - airPLS(y1)  # corrected values
    c2 = y2 - airPLS(y2)  # with baseline removed
    print
    'Plotting results'
    fig, ax = plt.subplots(nrows=2, ncols=1)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }

    #ax[0].plot(x, y1, '-k')
    ax[0].plot(waveLength, c1, '-r')
    ax[0].plot(waveLength,y1,'-k')
    labels0 = ax[0].get_xticklabels() + ax[0].get_yticklabels()

    ax[0].tick_params(labelsize=15)
    [label0.set_fontname('normal') for label0 in labels0]
    [label0.set_fontstyle('normal') for label0 in labels0]
    ax[0].set_title('Linear baseline',font2)
    ax[1].plot(waveLength, y2, '-k')
    ax[1].plot(waveLength, c2, '-r')
    ax[1].set_title('sinusoidal baseline',font2)
    plt.tick_params(labelsize=15)

    labels1 = ax[1].get_xticklabels() + ax[1].get_yticklabels()
    [label.set_fontname('normal') for label in labels1]
    [label.set_fontstyle('normal') for label in labels1]
    plt.show()
    print
    'Done!'
