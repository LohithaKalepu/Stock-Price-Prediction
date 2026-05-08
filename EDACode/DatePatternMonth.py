import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.patches as mpatches

tickers    = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"
end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")
colors     = ["#378ADD", "#1D9E75", "#BA7517", "#D85A30", "#7F77DD"]
months     = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# -----------------------------
# Data
# -----------------------------
raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
monthly_prices  = raw.resample("ME").last()
monthly_returns = monthly_prices.pct_change().dropna() * 100
monthly_returns["MonthNum"] = monthly_returns.index.month

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(22, 7))

n_tickers   = len(tickers)
group_width = 0.8
box_width   = group_width / n_tickers
offsets     = np.linspace(-group_width/2 + box_width/2,
                          group_width/2 - box_width/2,
                          n_tickers)

for i, (ticker, color) in enumerate(zip(tickers, colors)):
    data_by_month = [
        monthly_returns.loc[monthly_returns["MonthNum"] == m, ticker].dropna().values
        for m in range(1, 13)
    ]

    positions = np.arange(12) + offsets[i]

    # --- Boxplots ---
    ax.boxplot(
        data_by_month,
        positions=positions,
        widths=box_width * 0.85,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color=color, linewidth=1.0, alpha=0.8),
        capprops=dict(color=color, linewidth=1.2),
        boxprops=dict(facecolor=color, alpha=0.75, linewidth=0.5),
    )

    # --- Mean + 95% CI error bars ---
    for j, grp in enumerate(data_by_month):
        if len(grp) < 2:
            continue

        mean = grp.mean()
        ci   = stats.sem(grp) * stats.t.ppf(0.975, df=len(grp) - 1)

        ax.errorbar(
            positions[j], mean,
            yerr=ci,
            fmt="D",                 # diamond marker
            color="black",
            markersize=3.5,
            capsize=3,
            linewidth=1.1,
            zorder=5
        )

# -----------------------------
# Styling
# -----------------------------
for m in range(0, 12, 2):
    ax.axvspan(m - 0.5, m + 0.5, color="gray", alpha=0.06, zorder=0)

ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_xticks(range(12))
ax.set_xticklabels(months, fontsize=11)
ax.set_ylabel("Monthly Return (%)", fontsize=11)
ax.set_xlim(-0.5, 11.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)

ax.set_title(
    "Monthly Returns by Month of Year — All Stocks (2007–Present)",
    fontsize=13, fontweight="bold", pad=12
)

# Legend
legend_patches = [mpatches.Patch(facecolor=c, alpha=0.75, label=t)
                  for t, c in zip(tickers, colors)]

mean_patch = mpatches.Patch(facecolor="none", edgecolor="none",
                            label="◆ Mean ± 95% CI")

ax.legend(handles=legend_patches + [mean_patch],
          loc="lower left", framealpha=0.5, fontsize=10)

plt.tight_layout()
plt.savefig("monthly_returns_grouped_errorbars.png", dpi=150, bbox_inches="tight")
plt.show()
