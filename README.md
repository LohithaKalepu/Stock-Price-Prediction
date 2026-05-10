
## Project Overview

This project builds an end-to-end pipeline that:
- Collects and engineers financial features from 20 years of OHLCV data (Yahoo Finance)
- Generates daily sentiment scores from financial news using FinBERT (MarketAUX API, 2022–2025)
- Trains and evaluates ML and DL models using a rolling 5-year window strategy
- Tests early fusion and late fusion strategies for combining financial and sentiment signals
- Selects the best ensemble model based on Directional Accuracy and Sharpe Ratio

**Best Model:** TFT + TCN + XGBoost (Early Fusion, Simple Average)  
**Directional Accuracy:** 56.4% | **Sharpe Ratio:** 1.69

---
## Team
Lohitha Kalepu, Dylan Toriello, Dhruv Oza, Ryan Ryu, Emmanuelna Surpris

---

## Repository Structure
(Still Working)
