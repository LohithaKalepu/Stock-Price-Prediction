import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

tickers    = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"
end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")
colors     = ["#378ADD", "#1D9E75", "#BA7517", "#D85A30", "#7F77DD"]

raw     = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
returns = raw.pct_change().dropna()

fig, axes = plt.subplots(5, 2, figsize=(14, 22))

for i, (ticker, color) in enumerate(zip(tickers, colors)):
    data = returns[ticker].dropna()

    # --- Histogram ---
    ax_hist = axes[i, 0]

    ax_hist.hist(data, bins=120, color=color, alpha=0.7,
                 density=True, label="Actual returns")

    # Overlay a normal distribution with same mean and std for comparison
    x = np.linspace(data.min(), data.max(), 300)
    normal_curve = stats.norm.pdf(x, data.mean(), data.std())
    ax_hist.plot(x, normal_curve, color="black", linewidth=1.5,
                 linestyle="--", label="Normal distribution")

    # Annotate with skewness and kurtosis
    skew = data.skew()
    kurt = data.kurtosis()  # excess kurtosis (normal = 0)
    ax_hist.text(0.97, 0.95,
                 f"Skewness: {skew:.3f}\nExcess kurtosis: {kurt:.3f}",
                 transform=ax_hist.transAxes, fontsize=8.5,
                 ha="right", va="top",
                 bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))

    ax_hist.set_title(f"{ticker} — Return Distribution", fontsize=11,
                      fontweight="bold", color=color)
    ax_hist.set_xlabel("Daily Return", fontsize=9)
    ax_hist.set_ylabel("Density", fontsize=9)
    ax_hist.legend(fontsize=8)
    ax_hist.spines[["top", "right"]].set_visible(False)

    # --- QQ Plot ---
    ax_qq = axes[i, 1]

    # Compute theoretical quantiles vs actual quantiles
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    ax_qq.scatter(osm, osr, color=color, alpha=0.4, s=3, label="Return quantiles")

    # Reference line (what perfect normality looks like)
    x_line = np.array([min(osm), max(osm)])
    ax_qq.plot(x_line, slope * x_line + intercept, color="black",
               linewidth=1.5, linestyle="--", label="Normal reference line")

    ax_qq.set_title(f"{ticker} — QQ Plot", fontsize=11,
                    fontweight="bold", color=color)
    ax_qq.set_xlabel("Theoretical quantiles", fontsize=9)
    ax_qq.set_ylabel("Sample quantiles", fontsize=9)
    ax_qq.legend(fontsize=8, markerscale=3)
    ax_qq.spines[["top", "right"]].set_visible(False)

    # --- Shapiro-Wilk normality test ---
    # Shapiro-Wilk becomes unreliable above ~5000 samples so we subsample
    sample = data.sample(min(len(data), 4999), random_state=42)
    _, p_val = stats.shapiro(sample)
    p_label  = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
    ax_qq.text(0.03, 0.95, f"Shapiro-Wilk p={p_label}",
               transform=ax_qq.transAxes, fontsize=8.5,
               ha="left", va="top",
               bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))

fig.suptitle("Return Distributions — Histogram & QQ Plot (2007–Present)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("return_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
