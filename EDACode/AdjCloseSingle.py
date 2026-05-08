import yfinance as yf
import matplotlib.pyplot as plt

tickers = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"

df = yf.download(tickers, start=start_date, auto_adjust=False, progress=False)
adj_close = df["Adj Close"]

for ticker in tickers:
    plt.figure(figsize=(10, 5))
    plt.plot(adj_close.index, adj_close[ticker])
    plt.title(f"{ticker} Adjusted Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

