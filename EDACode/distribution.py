import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# Settings
# -----------------------------
tickers = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# -----------------------------
# Download prices
# -----------------------------
raw = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True
)["Close"]

# -----------------------------
# Use log returns for ML
# -----------------------------
returns = np.log(raw / raw.shift(1)).dropna()

# Optional: standardized returns for easier cross-stock comparison
z_returns = (returns - returns.mean()) / returns.std()

# -----------------------------
# Plot 1: Histogram + KDE + Normal Curve
# -----------------------------
fig, axes = plt.subplots(len(tickers), 2, figsize=(14, 22))

for i, ticker in enumerate(tickers):
    data = returns[ticker].dropna()

    # Left: histogram + KDE + normal curve
    ax1 = axes[i, 0]

    # Histogram
    ax1.hist(data, bins=80, density=True, alpha=0.6, label="Log returns")

    # KDE
    kde = stats.gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 400)
    ax1.plot(x, kde(x), linewidth=2, label="KDE")

    # Normal curve with same mean/std
    mu, sigma = data.mean(), data.std()
    normal_pdf = stats.norm.pdf(x, mu, sigma)
    ax1.plot(x, normal_pdf, linestyle="--", linewidth=2, label="Normal fit")

    skew = data.skew()
    kurt = data.kurtosis()

    ax1.set_title(f"{ticker} — Distribution of Log Returns")
    ax1.set_xlabel("Log Return")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.text(
        0.97, 0.95,
        f"Skew: {skew:.3f}\nExcess Kurtosis: {kurt:.3f}",
        transform=ax1.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: QQ plot
    ax2 = axes[i, 1]
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title(f"{ticker} — QQ Plot")
    ax2.spines[["top", "right"]].set_visible(False)

plt.suptitle("Return Distribution Diagnostics for ML", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("distribution_plots_ml.png", dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------
# Plot 2: Boxplots for outliers
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot([returns[t].dropna() for t in tickers], labels=tickers, showfliers=True)
ax.set_title("Boxplots of Log Returns")
ax.set_ylabel("Log Return")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("boxplots_log_returns.png", dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------
# Plot 3: Standardized returns comparison
# -----------------------------
fig, axes = plt.subplots(len(tickers), 1, figsize=(10, 16), sharex=False)

for i, ticker in enumerate(tickers):
    data = z_returns[ticker].dropna()
    ax = axes[i]

    ax.hist(data, bins=80, density=True, alpha=0.7)
    ax.set_title(f"{ticker} — Standardized Return Distribution")
    ax.set_xlabel("Z-scored Log Return")
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("standardized_return_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
