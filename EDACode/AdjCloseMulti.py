import yfinance as yf
import matplotlib.pyplot as plt

tickers = ["GOOGL", "PFE", "XOM", "TSLA", "JPM"]
start_date = "2007-01-01"

df = yf.download(tickers, start=start_date, auto_adjust=False, progress=False)
adj_close = df["Adj Close"]

plt.figure(figsize=(12, 6))

for ticker in tickers:
    plt.plot(adj_close.index, adj_close[ticker], label=ticker)

plt.yscale("log")

plt.title("Stock Prices (Log Scale)")
plt.xlabel("Date")
plt.ylabel("Price (log scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
