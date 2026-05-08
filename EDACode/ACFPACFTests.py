import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

tickers    = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"
end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")
colors     = ["#378ADD", "#1D9E75", "#BA7517", "#D85A30", "#7F77DD"]

raw     = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
returns = raw.pct_change().dropna()

fig, axes = plt.subplots(5, 2, figsize=(14, 22))

for i, (ticker, color) in enumerate(zip(tickers, colors)):
    data = returns[ticker].dropna()

    # --- ACF ---
    ax_acf = axes[i, 0]
    plot_acf(data, lags=40, ax=ax_acf, color=color,
             vlines_kwargs={"colors": color},
             alpha=0.05)                        # 95% confidence bands

    ax_acf.set_title(f"{ticker} — ACF", fontsize=11,
                     fontweight="bold", color=color)
    ax_acf.set_xlabel("Lag (days)", fontsize=9)
    ax_acf.set_ylabel("Autocorrelation", fontsize=9)
    ax_acf.axhline(0, color="black", linewidth=0.8)
    ax_acf.spines[["top", "right"]].set_visible(False)

    # --- PACF ---
    ax_pacf = axes[i, 1]
    plot_pacf(data, lags=40, ax=ax_pacf, color=color,
              vlines_kwargs={"colors": color},
              method="ywm",                     # Yule-Walker method, most stable for returns
              alpha=0.05)

    ax_pacf.set_title(f"{ticker} — PACF", fontsize=11,
                      fontweight="bold", color=color)
    ax_pacf.set_xlabel("Lag (days)", fontsize=9)
    ax_pacf.set_ylabel("Partial Autocorrelation", fontsize=9)
    ax_pacf.axhline(0, color="black", linewidth=0.8)
    ax_pacf.spines[["top", "right"]].set_visible(False)

    ax_acf.set_xlim(left=0.5)
    ax_acf.set_ylim(-0.1, 0.1)   # zoom in — adjust range if bars still clip

    ax_pacf.set_xlim(left=0.5)
    ax_pacf.set_ylim(-0.1, 0.1)

fig.suptitle("ACF & PACF of Daily Returns (2007–Present)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("acf_pacf.png", dpi=150, bbox_inches="tight")
plt.show()
