# ML Systematic Trading Engine — Project Summary for Presentation

## Project Overview
This is an end-to-end machine learning systematic trading system for BTCUSDT (Bitcoin/USD) futures, built as a class project. The system covers the full pipeline: data collection, feature engineering, model discovery, backtesting, and live deployment on a real exchange. The project is implemented in Python using PyTorch (neural networks), Polars (data manipulation), and the Binance Futures API.

---

## Project Structure

```
ML-Systematic-Trading-Engine/
├── research.py                  # Core research library (shared utilities)
├── models.py                    # PyTorch model definitions
├── binance.py                   # Binance constants (fee rates, etc.)
├── live_exchange.py             # Live Binance exchange adapter (REST + WebSocket)
├── live_strategy.py             # Live trading strategy + real-time execution loop
├── analyze_logs.py              # Post-trade log analysis tool
├── model_discovery/
│   ├── video1.ipynb             # Original tutorial notebook (creator's 12h model)
│   ├── model_discovery_8h.ipynb # 8-hour interval model benchmarking
│   ├── model_discovery_12h.ipynb# 12-hour interval model benchmarking
│   └── model_discovery_24h.ipynb# 24-hour (daily) interval model benchmarking ← WINNER
├── trading_strategy/
│   ├── trading_strat_8h_2lag.ipynb   # Backtester for 8h model
│   ├── trading_strat_12h_lag3.ipynb  # Backtester for 12h model
│   └── trading_strat_1d_lag3.ipynb   # Backtester for 1d model ← BEST NET RETURNS
├── data_scripts/
│   ├── generate_8h_ohlc.py
│   ├── generate_12h_ohlc.py
│   └── generate_24h_ohlc.py
└── historical_data/
    ├── BTCUSDT_12h_ohlc.csv         # Original dataset (Oct 2024 – Oct 2025)
    ├── BTCUSDT_8h_ohlc_updated.csv  # Updated dataset (May 2025 – May 2026)
    ├── BTCUSDT_12h_ohlc_updated.csv
    └── BTCUSDT_1d_ohlc_updated.csv
```

---

## Data Pipeline

Raw tick data (1-minute BTCUSDT trade data stored as daily .parquet files in a local cache) is aggregated into OHLC (Open-High-Low-Close) candles using a custom parallel processing pipeline. Three time intervals are used: 8h, 12h, and 1d (daily). All datasets cover approximately 1 year of data ending in May 2026.

**Key discovery:** An earlier dataset (Oct 2024 – Oct 2025) captured a strong Bitcoin bull run and led to overfitted models with unrealistically high Sharpe ratios (10.9). The updated datasets (May 2025 – May 2026) reflect a more volatile, realistic market regime and produce models with genuinely robust out-of-sample performance.

---

## Model Architecture

**Model:** `LinearModel` — a single-layer PyTorch linear regression model (essentially `y = w·X + b`).

```python
class LinearModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    def forward(self, x):
        return self.linear(x)
```

**Why linear?** Interpretability. A linear model with one weight is trivially explainable — we know exactly why every trade was made. This is a deliberate design choice over black-box neural networks.

**Features:** Log returns of the closing price at various lags:
- `close_log_return_lag_N` = `log(close[t-N] / close[t-N-1])`

**Target:** `close_log_return` = `log(close[t] / close[t-1])` (next interval's return)

**Loss function:** L1 Loss (Mean Absolute Error) — chosen over MSE because it is robust to large price spike outliers in crypto data.

**Train/test split:** 75% train, 25% test (time-series split, no shuffling to prevent look-ahead bias).

---

## Model Discovery Process

The `research.benchmark_linear_models()` function exhaustively tests all combinations of lag features (up to a configurable maximum) across all time intervals, returning a ranked DataFrame sorted by annualized Sharpe ratio.

**Models evaluated:** 8h, 12h, and 1d (daily) intervals with up to 4 lags. All combinations of 1, 2, and 3 features were tested. Models were evaluated on out-of-sample test data.

**Key insight discovered:** Longer time intervals produce significantly better NET returns after transaction fees. An 8h model trades ~1,095 times per year (3×/day), while a 1d model trades only ~90 times per year. With Binance charging 0.045% per taker order, the fee reduction alone is the difference between a losing and winning strategy.

---

## Model Results Summary

| Model | Interval | Features | Sharpe | Gross Return | Net Return |
|---|---|---|---|---|---|
| 8h 2-lag | 8h | lag_2 + lag_3 | 2.1 | 1.29x | Negative after fees |
| 12h lag3 | 12h | lag_3 | 1.69 | 1.25x | ~11% constant sizing |
| **1d lag3 ← WINNER** | **1d** | **lag_3** | **3.32** | **1.58x** | **34–42% net** |

**Winner details (`1d_lag3_model.pth`):**
- Weight: [-0.02642301], Bias: -0.000109
- 90 trades over the 3-month test period (~Feb – May 2026)
- Win rate: 54.4%
- Max drawdown: -14.5%
- Compound net return (after taker fees): **42%**
- Constant sizing net return (after taker fees): **34%**
- Annualized Sharpe: **3.32** (above Renaissance Medallion Fund's ~2.0 average)

**Model interpretation:** The negative weight is the signature of a **mean reversion** strategy. If Bitcoin had a large positive return 3 days ago, the model predicts a small pullback today, and vice versa. This is a well-documented market microstructure pattern in daily crypto data.

---

## Backtesting Framework

The vectorized backtester (`trading_strat_1d_lag3.ipynb`) computes the full year's trades in a single Polars DataFrame operation:

1. **Signal generation:** `dir_signal = sign(y_hat)` → +1 (long) or -1 (short)
2. **Trade return:** `trade_log_return = close_log_return × dir_signal`
3. **Two sizing strategies:**
   - **Compound sizing:** reinvest gains, bet a fixed % of growing portfolio
   - **Constant sizing:** trade fixed notional ($100) every time
4. **Transaction costs:** Taker fee (0.045%) applied to both entry and exit value
5. **Equity curve:** cumulative sum of net PnL after fees

The backtester includes Maker fee curves as well (0.02%), which would improve returns further if limit orders were used in live trading.

---

## Live Trading System

The live system consists of two files:

### `live_exchange.py`
- `BinanceFuturesExchange`: Authenticated REST client. Wraps Binance USD-M Futures API with HMAC-SHA256 request signing for balance queries, position lookups, and market order execution. Supports both testnet (paper trading) and live modes.
- `BinanceMarkPriceWebsocket`: Persistent WebSocket connection receiving the Binance mark price every second. Includes auto-reconnect logic. Runs on a background thread.
- `run_live_strategy_loop()`: Glue function. Every price tick → calls `strategy.on_tick()` → executes any returned orders → logs all trades to `stats/trading_log.csv`.

### `live_strategy.py`
- `IntervalFeatureExtractor`: On first tick, fetches historical daily candles from Binance public REST API and computes log returns. Caches them and refreshes automatically when the current candle expires. Supports specifying exactly which lags to return (e.g., `lags=[3]` gives only lag_3 as input to the model).
- `LiveTakerStrat`: Runs the PyTorch model on each price tick, decides whether to flip position direction. Critically, it skips re-execution if the model's predicted direction matches the current open position — preventing thousands of unnecessary trades per day.
- `main()`: Configures and launches the live bot. Model swapping is trivial: change `model_lags = [3]` to e.g. `[2, 3]` and update the .pth file path.

**Configuration (current live deployment):**
```python
model_lags = [3]
model = LinearModel(len(model_lags))  # LinearModel(1)
model.load_state_dict(torch.load('trading_strategy/1d_lag3_model.pth'))
extractor = IntervalFeatureExtractor(symbol='BTCUSDT', interval='1d', lags=model_lags)
scale_factor = Decimal('0.1')  # trade 10% of balance per signal
```

**Deployment:** The strategy is running live on a cloud server connected to Binance Futures (using the futures testnet for paper trading). The system uses Binance futures which supports both long and short positions, allowing the full mean-reversion strategy to execute on both bullish and bearish signals.

---

## Key Lessons Learned

1. **Simpler models generalize better.** A 1-parameter linear model outperformed every complex model attempted. Occam's Razor in action.
2. **Transaction fees kill short-interval strategies.** The 8h model was profitable in gross terms but negative after fees. Reducing trade frequency from 8h to 1d turned a losing strategy into a 34–42% net winner.
3. **Regime awareness matters.** Models trained on the 2024-2025 bull market completely failed on 2025-2026 out-of-sample data. Always validate on recent data.
4. **Volatility drag is real.** For this model, constant-size position sizing outperformed compound sizing because the model's occasional large losses punish compounded positions disproportionately.
5. **The infrastructure is model-agnostic.** The entire research, backtesting, and live execution pipeline generalizes to any asset, any time interval, and any PyTorch model — by changing just a few configuration lines.
