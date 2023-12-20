"""
    Hurst Exponent
    author: Jakub Polec (@jakubpolec)
    date: 2023-12-15
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from pprint import pprint

def angular_decomposition(price_data, amplitude_threshold=2, frequency_threshold=0.1):
    """
    Decompose price time series into trend and noise components.
    
    Args:
    - price_data (Pandas Series): Closing prices over time.
    - amplitude_threshold (float): Cutoff for separating high/low frequencies.
    - frequency_threshold (float): Cutoff for separating trends/noise.
        
    Returns: 
    - trends (Pandas Series): Trend component of prices.
    - noise (Pandas Series): Noise component of prices.
    """
    
    # Convert Pandas Series to NumPy array
    price_array = price_data.to_numpy()

    # Apply Fourier Transform
    fft_prices = fft(price_array)
    freqs = fftfreq(len(price_array))
    
    # Initialize components
    trend_component = np.zeros_like(fft_prices)
    noise_component = np.zeros_like(fft_prices)
    
    # Apply thresholds to filter frequencies
    for i, freq in enumerate(freqs):
        if np.abs(fft_prices[i]) > amplitude_threshold and np.abs(freq) < frequency_threshold:
            trend_component[i] = fft_prices[i]
        else:
            noise_component[i] = fft_prices[i]
    
    # Inverse Fourier Transform to get time domain components
    trends = ifft(trend_component).real
    noise = ifft(noise_component).real
    
    return pd.Series(trends, index=price_data.index), pd.Series(noise, index=price_data.index)


def analyze_components(trends, noise, original_data):
    """
    Analyze the strength of trends and noise level in the decomposed components.

    Args:
        trends (Pandas Series): Trend component of the time series.
        noise (Pandas Series): Noise component of the time series.
        original_data (Pandas Series): Original time series data.

    Returns:
        dict: Analysis results including trend strength, noise level, and Sharpe ratio.
    """
    trend_strength = trends.abs().sum() / original_data.abs().sum()
    noise_level = noise.std() / original_data.std()
    sharpe_ratio = trends.mean() / trends.std() * np.sqrt(252)  # Assuming daily data for annualized Sharpe ratio

    return {
        "trend_strength": trend_strength,
        "noise_level": noise_level,
        "sharpe_ratio": sharpe_ratio
    }


def decompose_hurst(hurst):
    """
    Decompose Hurst exponent into short-term and long-term components.

    Args:
        hurst (float): Hurst exponent value.

    Returns:
        tuple: short-term and long-term components.
    """
    short_term = 0.5 * (1 + np.sin((hurst - 0.5) * np.pi / 2))
    long_term = 0.5 * (1 - np.cos((hurst - 0.5) * np.pi / 2))
    return short_term, long_term


def cluster_hurst(hurst_values, n_clusters=3):
    """
    Cluster Hurst values into regimes using K-Means algorithm.

    Args:
        hurst_values (array-like): Array of Hurst exponent values.
        n_clusters (int): Number of clusters.

    Returns:
        array: Cluster labels for each Hurst value.
    """
    model = KMeans(n_clusters=n_clusters)
    model.fit(np.array(hurst_values).reshape(-1, 1))
    return model.labels_


def cluster_hurst_behaviors(time_series, window_size, n_clusters):
    """
    Cluster Hurst behaviors using rolling windows.
    
    Args:
        time_series (Pandas Series): Time series data.
        window_size (int): Size of rolling window.
        n_clusters (int): Number of clusters.
    
    Returns:
        array: Cluster labels for each Hurst behavior.
    """
    # Initialize array of Hurst values
    hurst_values = []
    pad_length = window_size - 1  # Length to pad the clusters array

    # Calculate Hurst exponent over rolling windows
    for i in range(len(time_series) - window_size + 1):
        window = time_series[i:i+window_size]
        hurst_exp = hurst_exponent(window)
        hurst_values.append(hurst_exp)

    # Cluster the Hurst values
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(np.array(hurst_values).reshape(-1, 1))

    # Pad the beginning of the clusters array
    padded_clusters = np.pad(clusters, (pad_length, 0), 'constant', constant_values=(0,))

    return padded_clusters


def cluster_hurst_behaviors_de_noised(trend_data, window_size, n_clusters):
    hurst_values = []
    pad_length = window_size - 1  # Length to pad the clusters array

    # Calculate Hurst exponent over rolling windows on the trend data
    for i in range(len(trend_data) - window_size + 1):
        window = trend_data[i:i+window_size]
        hurst_exp = hurst_exponent(window)
        hurst_values.append(hurst_exp)

    # Cluster the Hurst values
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(np.array(hurst_values).reshape(-1, 1))

    # Pad the beginning of the clusters array
    padded_clusters = np.pad(clusters, (pad_length, 0), 'constant', constant_values=(0,))

    return padded_clusters


def hurst_exponent(time_series, min_lag=2, max_lag=100, regression='linear'):
    """
    Calculate the Hurst exponent for a time series.

    :param time_series: List or numpy array of time series data
    :param min_lag: Minimum lag value (default is 2)
    :param max_lag: Maximum lag value (default is 100)
    :param regression: Type of regression ('linear' or 'logarithmic')
    :return: Hurst exponent
    """

    # Ensure time_series is a numpy array
    time_series = np.array(time_series)

    # Generate lags
    lags = range(min_lag, min(max_lag, len(time_series)//2))

    # Calculate the array of variances of lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # Perform regression
    log_lags = np.log(np.array(list(lags)))
    log_tau = np.log(np.array(tau))

    if regression == 'linear':
        model = LinearRegression()
        model.fit(log_lags.reshape(-1, 1), log_tau)
        return model.coef_[0] * 2.0
    elif regression == 'logarithmic':
        poly = np.polyfit(log_lags, log_tau, 1)
        return poly[0] * 2.0
    else:
        raise ValueError("Invalid regression type. Use 'linear' or 'logarithmic'.")

def bayesian_update(time_series, initial_beliefs):
    with pm.Model() as model:
        # Define the prior distributions based on initial beliefs
        # Example: half_life as a normal distribution
        half_life = pm.Normal('half_life', mu=initial_beliefs['half_life_mean'], sigma=initial_beliefs['half_life_sd'])
        
        # Define the likelihood function based on your time series data
        # This is a placeholder and should be replaced with a specific likelihood function
        likelihood = pm.Normal('likelihood', mu=half_life, sd=1, observed=time_series)
        
        # Perform the Bayesian update
        trace = pm.sample(1000, return_inferencedata=False)

    updated_beliefs = {
        'half_life_mean': np.mean(trace['half_life']),
        'half_life_sd': np.std(trace['half_life'])
        # Include other parameters as needed
    }

    return updated_beliefs

def bayesian_hurst(hurst_data):
    with pm.Model() as model:
        # Define priors for the parameters
        half_life = pm.Exponential('half_life', 1.)
        periodicity = pm.DiscreteUniform('periodicity', lower=1, upper=10)
        
        # Placeholder for likelihood function - this needs to be defined based on your data and model
        likelihood = pm.Normal('likelihood', mu=half_life, sd=1, observed=hurst_data)
        
        # Perform the Bayesian update
        trace = pm.sample(2000, return_inferencedata=False)
    
    return trace

# Generate synthetic time series data
np.random.seed(0)
time = pd.date_range(start='2020-01-01', periods=200, freq='D')
prices = np.cumsum(np.random.randn(200)) + 100  # Random walk with drift
price_data = pd.Series(prices, index=time)

# Assuming angular_decomposition function is already defined
trends, noise = angular_decomposition(price_data)

# Assuming analyze_components function is already defined
analysis = analyze_components(trends, noise, price_data)

# Assuming hurst_exponent function is already defined
hurst_exp = hurst_exponent(price_data)

# Assuming cluster_hurst_behaviors function is already defined
clusters = cluster_hurst_behaviors(price_data, window_size=20, n_clusters=3)

# Assuming cluster_hurst_behaviors_de_noised function is already defined
clusters_trend = cluster_hurst_behaviors_de_noised(trends, window_size=20, n_clusters=3)

# Ensure the lengths match
assert len(time) == len(price_data) == len(clusters)
assert len(time) == len(trends) == len(clusters_trend)

# Creating a subplot for each analysis
fig, axs = plt.subplots(4, 1, figsize=(6, 12))

# Plotting each analysis
# Plot 1: Original Data
axs[0].plot(price_data, label='Synthetic Price Data')
axs[0].set_title('Synthetic Time Series Data')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Price')
axs[0].legend()

# Plot 2: Angular Decomposition
axs[1].plot(price_data, label='Original Data')
axs[1].plot(trends, label='Trend Component', alpha=0.7)
axs[1].plot(noise, label='Noise Component', alpha=0.7)
axs[1].set_title('Angular Decomposition')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Price')
axs[1].legend()

# Plot 3: Clustered Regimes
sc1 = axs[2].scatter(time, price_data, c=clusters, cmap='viridis')
axs[2].set_title('Clustered Regimes based on Rolling Window Hurst Exponent')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Price')
fig.colorbar(sc1, ax=axs[2], label='Cluster')

# Plot 4: Clustered Regimes on De-noised Data
sc2 = axs[3].scatter(time, trends, c=clusters_trend, cmap='viridis')
axs[3].set_title('Clustered Regimes on De-noised Data (Trend Component)')
axs[3].set_xlabel('Date')
axs[3].set_ylabel('De-noised Price (Trend)')
fig.colorbar(sc2, ax=axs[3], label='Cluster')

# Adjust layout
plt.tight_layout()
plt.show()