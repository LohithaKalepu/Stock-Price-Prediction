import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('feature_selected_dataset (1).csv')

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df.set_index('date')

# EWMA volatility
plt.figure(figsize=(14, 5))
plt.plot(df.index, df['ewm_vol_10'], label='EWMA Volatility')

years = df.index.year.unique()

for y in years:
    plt.axvline(pd.Timestamp(f"{y}-01-01"), color='gray', alpha=0.3)

plt.title("EWMA Volatility Over Time (Yearly Regimes)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()


#Smoothed volatility (rolling)
plt.figure(figsize=(14, 5))
df['ewm_vol_10'].plot(label='EWMA Volatility')

plt.title("EWMA Volatility Trend (Rolling)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()
