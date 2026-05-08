import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

tickers    = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"
end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")
days       = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
colors     = ["#378ADD", "#1D9E75", "#BA7517", "#D85A30", "#7F77DD"]

raw     = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
returns = raw.pct_change().dropna().abs() * 100
returns["DayName"] = returns.index.day_name()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, (ticker, color) in enumerate(zip(tickers, colors)):
    ax = axes[i]

    groups = [
        returns.loc[returns["DayName"] == day, ticker].dropna().values
        for day in days
    ]

    # --- Boxplot ---
    ax.boxplot(
        groups,
        positions=range(len(days)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=color, linewidth=1.2, alpha=0.8),
        capprops=dict(color=color, linewidth=1.5),
        boxprops=dict(facecolor=color, alpha=0.75, linewidth=0.5),
    )

    # --- Mean + 95% CI error bars ---
    for j, grp in enumerate(groups):
        mean = grp.mean()
        ci   = stats.sem(grp) * stats.t.ppf(0.975, df=len(grp) - 1)

        ax.errorbar(
            j, mean, yerr=ci,
            fmt="D", color="black",
            markersize=3.5,
            capsize=4,
            linewidth=1.2,
            zorder=5
        )

    # --- Styling ---
    ax.set_title(ticker, fontsize=13, fontweight="bold", color=color, pad=10)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([d[:3] for d in days], fontsize=10)
    ax.set_ylabel("Absolute Daily Return (%)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)

axes[5].set_visible(False)

# Updated legend (no stats)
fig.text(0.72, 0.18,
         "◆ = mean  |  error bars = 95% CI",
         fontsize=8.5, va="top",
         bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))

fig.suptitle("Volatility by Day of Week (2007–Present)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("volatility_by_day_individual.png", dpi=150, bbox_inches="tight")
plt.show()
