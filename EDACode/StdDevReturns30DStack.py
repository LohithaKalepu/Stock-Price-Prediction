import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"

# Download data
df = yf.download(tickers, start=start_date, auto_adjust=False, progress=False)

# Get adjusted close prices
adj_close = df["Adj Close"]

# Compute returns
returns = adj_close.pct_change()

# Compute 30-day rolling std
rolling_std_30 = returns.rolling(window=30).std()

# Create 5 stacked plots
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 18), sharex=True)

for i, ticker in enumerate(tickers):
    axes[i].plot(rolling_std_30.index, rolling_std_30[ticker])
    axes[i].set_title(f"{ticker} 30-Day Rolling Volatility")
    axes[i].set_ylabel("Std Dev")
    axes[i].grid(True)

axes[-1].set_xlabel("Date")

plt.tight_layout()
plt.show()
