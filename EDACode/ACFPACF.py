import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

# --- DEFINE THESE FIRST ---
tickers = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]

start_date = "2007-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# --- DOWNLOAD DATA ---
raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
returns = raw.pct_change().dropna()

# --- ACF / PACF TEXT OUTPUT ---
max_lag = 20

for ticker in tickers:
    print("=" * 60)
    print(f"{ticker} — ACF & PACF (Daily Returns)")
    print("=" * 60)

    data = returns[ticker].dropna()

    acf_vals, acf_conf = acf(data, nlags=max_lag, alpha=0.05)
    pacf_vals, pacf_conf = pacf(data, nlags=max_lag, alpha=0.05, method="ywm")

    print("\nACF:")
    for lag in range(1, max_lag + 1):
        val = acf_vals[lag]
        lower, upper = acf_conf[lag]
        sig = "✓" if (val < lower or val > upper) else ""
        print(f"Lag {lag:2d}: {val: .5f}  [{lower:.5f}, {upper:.5f}] {sig}")

    print("\nPACF:")
    for lag in range(1, max_lag + 1):
        val = pacf_vals[lag]
        lower, upper = pacf_conf[lag]
        sig = "✓" if (val < lower or val > upper) else ""
        print(f"Lag {lag:2d}: {val: .5f}  [{lower:.5f}, {upper:.5f}] {sig}")

    print("\nLegend: ✓ = statistically significant\n")
