import yfinance as yf
import pandas as pd
import numpy as np

tickers = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Download adjusted prices
raw = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)["Close"]

# -------------------------
# Monthly median returns
# -------------------------
monthly_summary = {}

for ticker in tickers:
    s = raw[ticker].dropna()

    monthly_prices = s.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna() * 100

    monthly_summary[ticker] = [
        round(monthly_returns[monthly_returns.index.month == m].median(), 3)
        for m in range(1, 13)
    ]

df_monthly = pd.DataFrame(monthly_summary, index=months)

print("=" * 65)
print("MEDIAN MONTHLY RETURN (%) — 2007 to Present")
print("=" * 65)
print(df_monthly.to_string())

# -------------------------
# Daily median returns by weekday
# -------------------------
daily_summary = {}

for ticker in tickers:
    s = raw[ticker].dropna()

    daily_returns = s.pct_change().dropna() * 100

    daily_summary[ticker] = [
        round(daily_returns[daily_returns.index.day_name() == day].median(), 4)
        for day in days
    ]

df_daily = pd.DataFrame(daily_summary, index=days)

print()
print("=" * 65)
print("MEDIAN DAILY RETURN (%) BY WEEKDAY — 2007 to Present")
print("=" * 65)
print(df_daily.to_string())
