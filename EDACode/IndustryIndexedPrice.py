import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

#tickers = ['GOOGL', 'AAPL', 'MSFT', 'IBM', 'ORCL']
#tickers = ['PFE', 'JNJ', 'MRK', 'BMY', 'SNY']
#tickers = ['TSLA', 'F', 'TM', 'HMC', 'GM']
#tickers = ['JPM', 'BAC', 'C', 'WFC', 'GS']
tickers = ['XOM', 'CVX', 'BP', 'SHEL', 'TTE']
start_date = '2007-01-01'

data = yf.download(
    tickers,
    start=start_date,
    auto_adjust=True,
    progress=False
)

prices = data['Close'].copy()

# Make sure we have a DataFrame even for one ticker
if isinstance(prices, pd.Series):
    prices = prices.to_frame()

# Keep rows where at least one ticker has data
prices = prices.dropna(how='all')

# Normalize each ticker from its own first valid observation
normalized = pd.DataFrame(index=prices.index)

for ticker in prices.columns:
    s = prices[ticker].dropna()
    if not s.empty:
        normalized[ticker] = prices[ticker] / s.iloc[0] * 100

plt.figure(figsize=(12, 6))

for ticker in tickers:
    if ticker in normalized.columns:
        plt.plot(normalized.index, normalized[ticker], label=ticker)

plt.title('Indexed Performance')
plt.xlabel('Date')
plt.ylabel('Indexed Price (Base = 100)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.yscale('log')
plt.show()
