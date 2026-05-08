import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"

# Download once
df = yf.download(tickers, start=start_date, auto_adjust=False, progress=False)

adj_close = df["Adj Close"]

# Compute returns
returns = adj_close.pct_change()

# Compute 30-day rolling std
rolling_std_30 = returns.rolling(window=30).std()

import matplotlib.dates as mdates

# Plot one graph per stock
for ticker in tickers:
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_std_30.index, rolling_std_30[ticker])
    
    plt.title(f"{ticker} 30-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Std Dev")
    plt.grid(True)

    # Set x-axis to 6-month intervals
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.xticks(rotation=45)  # rotate for readability
    
    plt.tight_layout()
    plt.show()
    plt.close()
