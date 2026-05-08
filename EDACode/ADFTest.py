import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import adfuller

tickers    = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"
end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")

raw     = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
returns = raw.pct_change().dropna()

def run_adf(series, name, series_type):
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat, p_val, lags, _, critical_values, _ = result

    print(f"  ADF Statistic:   {adf_stat:.4f}")
    print(f"  p-value:         {p_val:.6f}")
    print(f"  Lags used:       {lags}")
    print(f"  Critical values:")
    for level, val in critical_values.items():
        print(f"    {level}: {val:.4f}")

    # Interpretation
    if p_val < 0.05:
        verdict = "STATIONARY ✓  (safe to use as model input)"
    else:
        verdict = "NON-STATIONARY ✗  (needs differencing before use)"
    print(f"  Verdict:         {verdict}")

print("=" * 60)
print("ADF TEST — RAW PRICES (expected: non-stationary)")
print("=" * 60)
for ticker in tickers:
    print(f"\n{ticker}:")
    run_adf(raw[ticker], ticker, "price")

print()
print("=" * 60)
print("ADF TEST — DAILY RETURNS (expected: stationary)")
print("=" * 60)
for ticker in tickers:
    print(f"\n{ticker}:")
    run_adf(returns[ticker], ticker, "return")
