import numpy as np
import matplotlib.pyplot as plt

# Setup_Parameters

S0 = 100  # Initial stck price
K = 110  # strike price
T = 1.0  # time to maturity (1 year)
r = 0.05  # risk-free rate
sigma = 0.2  # standard deviation/volatility
n_simulations = 10000 # number of simulations
n_steps = 252  # number of time steps (252 trading days)

# Simulate Price Paths

dt = T / n_steps
paths = np.zeros((n_simulations, n_steps + 1))
paths[:, 0] = S0

# Standard Normal random variables/numbers generator

for t in range(1, n_steps + 1):
    z = np.random.standard_normal(n_simulations)
    paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

# calculating option payoffs

payoffs = np.maximum(paths[:, -1] - K, 0)
option_price = np.exp(-r * T) * np.mean(payoffs)
print(f"Estimated option Price: {option_price:.2f}")

plt.plot(paths.T)
plt.title("Simulation Price Paths")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.show()