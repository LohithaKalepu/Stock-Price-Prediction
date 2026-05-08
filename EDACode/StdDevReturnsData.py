import yfinance as yf
import pandas as pd

tickers = ["GOOGL", "TSLA", "JPM", "XOM", "PFE"]
start_date = "2007-01-01"

# Download data
df = yf.download(tickers, start=start_date, progress=False)

# Use adjusted close prices
adj_close = df["Close"]

# Compute daily returns
returns = adj_close.pct_change().dropna()

# Compute standard deviation of daily returns
volatility = returns.std()

print(volatility)
