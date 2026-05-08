import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"

# Download data
df = yf.download(tickers, start=start_date, auto_adjust=False, progress=False)

# Get adjusted close prices
adj_close = df["Adj Close"]

# Compute daily returns
returns = adj_close.pct_change()

# Compute 30-day rolling standard deviation
rolling_std_30 = returns.rolling(window=30).std()

# Plot all 5 on one graph
plt.figure(figsize=(12, 6))

for ticker in tickers:
    plt.plot(rolling_std_30.index, rolling_std_30[ticker], label=ticker)

plt.title("30-Day Rolling Standard Deviation of Returns")
plt.xlabel("Date")
plt.ylabel("30-Day Rolling Std Dev")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
