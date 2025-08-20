import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Parameters (T, N, sigma, r, delta, lambda_0, q_max)
T = 1
N = 1000
sigma = 0.02
r = 0
lambda_0 = 1
q_max = 10
delta = 0.02
dt = T / N
########################################################
dW = np.random.normal(0, np.sqrt(dt), N)
S = np.cumsum(dW) * sigma
pnl = 0
q = 0

for t in range(1, N):
    p_b = S[t] - delta
    p_a = S[t] + delta
    bid = np.random.poisson(lambda_0 * np.exp(-delta))
    ask = np.random.poisson(lambda_0 * np.exp(delta))
    if bid and q < q_max:
        q += 1
        pnl -= p_b
    if ask and q > -q_max:
        q -= 1
        pnl += p_a
    if t % 100 == 0:  # Print every 100 iterations
        print(f"Current PnL: {pnl}")




