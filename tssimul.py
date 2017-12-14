''' Simulate ARIMA(p, 0, q) model '''

import argparse
import numpy as np
from scipy import stats

def simulate_eps(sigma, size, dist='normal', df=None):
    ''' Simulate innovation '''
    n_samples, length_sample = size

    if dist.startswith('n'):
        eps = np.random.standard_normal(size) * sigma
    elif dist.startswith('t'):
        eps = np.random.standard_t(df, size) / np.sqrt(df / (df - 2)) * sigma
    elif dist.startswith('exp'):
        eps = np.random.exponential(sigma, size) - sigma
    else:
        raise ValueError(f'Unrecognised distribution "{dist}"')

    return eps

def simulate_arima_given_innov(ar, ma, eps):
    ''' Simulate ARIMA '''
    n_samples, length_sample = eps.shape 

    order_p = len(ar)
    order_q = len(ma)
    assert order_p >= order_q

    samples = np.zeros_like(eps)
    samples_pre_innov = np.zeros_like(eps)

    for i in range(order_p, length_sample):
        samples[:, i] = (samples[:, (i - order_p):i].dot(ar[::-1]) + eps[:, i]
                         + eps[:, (i - order_q):i].dot(ma[::-1]))
    samples_pre_innov[:, order_p:] = samples[:, order_p:] - eps[:, order_p:]

    return np.concatenate([samples[:, None, :], samples_pre_innov[:, None, :],
                           eps[:, None, :]], axis=1)

def simulate_arima(ar, ma, sigma, size, dist='normal', df=None):
    ''' Simulate ARIMA '''
    eps = simulate_eps(sigma, size, dist, df)
    ts = simulate_arima_given_innov(ar, ma, eps)
    return ts

def simulate_sv(beta, sigma, intercept, size):
    ''' Simulate log-variance with AR1 '''
    n_samples, length_sample = size

    # Generate first the variational part with zero intercept
    # Finally add intercept to the entire array
    logvar = np.zeros(size)
    eps = np.random.standard_normal(size) * sigma
    
    logvar[:, 0] = 3 * eps[:, 0] 
    for i in range(1, length_sample):
        logvar[:, i] = beta * logvar[:, i - 1] + eps[:, i]
    
    var = np.exp(intercept + logvar)
    sv = np.random.normal(0, np.sqrt(var / 255), size)
    return sv, np.sqrt(var)

def simulate_rs(p0, p00, p10, mu, sigma, size):
    ''' Simulate regime-switching model '''
    n_samples, length_sample = size

    eps = [np.random.normal(m, s / np.sqrt(255), size)
           for m, s in zip(mu, sigma)]
    eps = np.stack(eps, axis=2)

    # Simulate regimes
    prob0 = np.zeros(size)
    regime = np.zeros(size)
    prob0[:, 0] = p0
    regime[:, 0] = 1 - np.random.binomial(1, p0, n_samples)
    for i in range(1, length_sample):
        prob0[:, i] = np.where(regime[:, i - 1] == 0, p00, p10)
        regime[:, i] = 1 - np.random.binomial(1, prob0[:, i], n_samples)

    ts = np.where(regime == 0, eps[:, :, 0], eps[:, :, 1])
    return ts, prob0, regime

if __name__ == '__main__':
    ar = [0.0868, 0.3667]
    ma = [-0.1150, -0.4068]
    sigma = .0112

    simulation = simulate_arima(5000, 1000, ar, ma, sigma)
    simulation_test = simulate_arima(500, 1000, ar, ma, sigma)

    np.savez_compressed('simulation', data=simulation)
    np.savez_compressed('simulation_test', data=simulation_test)
