
## Project Overview

This project builds an end-to-end pipeline that:
- Collects and engineers financial features from 20 years of OHLCV data (Yahoo Finance) 
- Generates daily sentiment scores from financial news using FinBERT (MarketAUX API, 2022–2025) 
- Trains and evaluates ML and DL models using a rolling window strategy 
- Tests early fusion and late fusion strategies for combining financial and sentiment signals 
- Selects the best ensemble model based on Directional Accuracy and Sharpe Ratio 

**Best Model:** TFT + CNN_LSTM + XGBoost (Early Fusion, Simple Average)  
**Directional Accuracy:** 56.4% | **Sharpe Ratio:** 1.69

---
## Team
Lohitha Kalepu, Dylan Toriello, Dhruv Oza, Ryan Ryu, Emmanuelna Surpris

---

## Repository Structure

NOTE: Though the earlier analyses like EDA and train/test split strategy trail and errors are performed on the 20 years stock data, the actual prediction scope of this project is restricted to GOOGL stock from 2022-2025 because of limited sentiment data-- we only selected GOOL stock across a small time period to ensure fair comparision in our pipeline. Further changes in this project will include 5 more stocks with more years worth of data. 

Please refer to the following structure to navigate through this repo--
```
EDACode
  ACFPACF.py                           # ACF and PACF plots for time series analysis
  ACFPACFTests.py                      # Statistical tests related to ACF/PACF
  ADFTest.py                           # Augmented Dickey-Fuller stationarity test
  AdjCloseMulti.py                     # Adjusted close price plots for multiple stocks
  AdjCloseSingle.py                    # Adjusted close price plot for a single stock
  DatePatternMonth.py                  # Monthly return/price pattern analysis
  DatePatternMonthText.py              # Prints median monthly and weekday returns as text tables
  DatePatternWeek.py                   # Weekly return/price pattern analysis
  DatePatternWeekVol.py                # Weekly volume pattern analysis
  IndustryIndexedPrice.py              # Indexed price comparison across industry stocks
  LagFeaturesTarget.txt                # Scatter plots of lagged return features vs next-day return target
  ReturnDistribution.py                # Return distribution plots
  StdDevReturns30DMulti.py             # 30-day rolling std dev of returns for multiple stocks
  StdDevReturns30DSingle.py            # 30-day rolling std dev of returns for a single stock
  StdDevReturns30DStack.py             # Stacked 30-day rolling std dev visualization
  StdDevReturnsData.py                 # Prints standard deviation of daily returns per ticker
  VolatilityTrend.py                   # Volatility trend over time
  distribution.py                      # Return distribution diagnostics: histograms, KDE, QQ plots, boxplots

FinDataCSVs
  ALL_ohlcv_long.csv                   # OHLCV data for all stocks in long format
  GOOGL_raw.parquet                    # Raw GOOGL stock data
  JPM_raw.parquet                      # Raw JPM stock data
  PFE_raw.parquet                      # Raw PFE stock data
  TSLA_raw.parquet                     # Raw TSLA stock data
  XOM_raw.parquet                      # Raw XOM stock data
  all_candidate_features.csv           # All engineered candidate features before selection
  engineered_stock_long.csv            # Engineered features in long format across stocks
  feature_selected_dataset.csv         # Dataset after feature selection
  final_selected_features.csv          # Final set of selected features used in modeling
  high_corr_feature_pairs.csv          # Highly correlated feature-pairs found at selection
  mutual_information_scores.csv        # Mutual information scores for feature ranking
  random_forest_feature_importance.csv # Feature importances from Random Forest selection

FinDataPreprocessing
  Stock_data_SplitAndModels.ipynb      # Baseline models (Linear Regression, Moving Average, ARIMA) with train/val/test split
  Stock_data_cleaning.ipynb            # Data cleaning pipeline for raw stock data
  Stock_data_collection.ipynb          # Data collection from financial APIs
  Stock_data_features.ipynb            # Feature engineering notebook
  Stock_data_fselection_eda.ipynb      # Feature selection and EDA notebook

GOOGL_Pipeline
  Early_Fusion_Google.ipynb            # Early fusion model combining financial and sentiment features
  Final_Ensemble_with_sentiment.ipynb  # Final ensemble model incorporating sentiment
  LSTM_36_combo_tuning_results.ipynb   # Grid search over 36 LSTM hyperparameter combinations across 15 rolling folds
  ML_With_Sentiment.ipynb              # ML models trained with sentiment features
  all_models_predictions.csv           # Compiled predictions from all models
  all_test_predictions.ipynb           # Merges ML and DL out-of-sample predictions into a single combined CSV
  dl_test_predictions.csv              # Test set predictions from deep learning models
  ensemble_predictions.csv             # Predictions from the ensemble model
  final_dl_GOOGL.ipynb                 # Final deep learning models for GOOGL
  final_ensemble.ipynb                 # Final ensemble model notebook
  final_features_GOOG22-25.ipynb       # Feature pipeline for GOOGL 2022–2025
  final_features_GOOGL_2022_2025.csv   # Final feature dataset for GOOGL 2022–2025
  final_late_fusion.ipynb              # Final late fusion model notebook
  final_ml_GOOGL.ipynb                 # Final ML models for GOOGL
  late_fusion_metrics.csv              # Evaluation metrics for late fusion models
  late_fusion_predictions.csv          # Predictions from late fusion models
  ml_test_predictions.csv              # Test set predictions from ML models
  rolling_window_results.csv           # Results from rolling window validation
  stock_sentiment_history.csv          # Original Stock Sentiment Score CSV

SentimentData
  Ingestion.py                         # Sentiment data ingestion script
  stock_sentiment_history(1).csv       # Final Sentiment Score CSV
```
