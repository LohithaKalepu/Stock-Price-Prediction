import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"

# Download data
df = yf.download(tickers, start=start_date, auto_adjust=False, progress=False)

adj_close = df["Adj Close"]

# Compute returns
returns = adj_close.pct_change()

# Add weekday (0=Mon, ..., 4=Fri)
returns["Weekday"] = returns.index.dayofweek

# Group by weekday and average  
weekly_avg = returns.groupby("Weekday").mean()

# Rename index
weekly_avg.index = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# Plot
weekly_avg.plot(kind="bar", figsize=(10,6))

plt.title("Average Daily Returns by Weekday")
plt.xlabel("Day of Week")
plt.ylabel("Average Return")
plt.grid(True)

plt.show()
