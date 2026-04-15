# ML Systematic Trading Engine

An educational, research-focused trading toolkit built around Binance futures trade data and time-series feature engineering.

This project is an independent reimplementation of the ideas presented in the memlabs video series [Let’s Build a Quant Trading Strategy](https://youtube.com/playlist?list=PLdsqLas3-Kg3fM8VgykXBJ2gRIFBpKHjY&si=hHZwOfD49Qy-AGZe). The goal is to create a clean, reproducible workflow for downloading market data, aggregating it into usable time-series bars, and experimenting with strategy research in Python.

## Project Goals

- Download and cache Binance futures trade data by day.
- Aggregate tick data into OHLC and custom time-series formats.
- Keep research utilities in one place so notebooks stay short and readable.
- Provide a simple base for testing indicators, features, and strategy ideas.

## Main Components

- [binance.py](binance.py) handles downloading, caching, and assembling Binance trade data.
- [research.py](research.py) contains reusable research utilities for time-series aggregation, OHLC feature building, and helper functions.
- [video1.ipynb](video1.ipynb) is the working notebook for experimentation and walkthroughs.
- [data/](data/) stores downloaded CSV trade files.
- [cache/](cache/) stores cached Parquet files for faster reloads.

## Requirements

The codebase is written for Python and uses these main libraries:

- `polars`
- `requests`
- `tqdm`
- `numpy`
- `torch`
- `altair`
- `matplotlib`

## Typical Workflow

1. Download raw trade data from Binance.
2. Cache the data locally as Parquet for repeatable experiments.
3. Convert the trades into OHLC bars or custom aggregated features.
4. Use the notebook to test ideas and visualize results.

## Example Usage

```python
from binance import download_trades, download_ohlc_timeseries

# Download the last 7 days of BTCUSDT trade data
trades = download_trades("BTCUSDT", 7, return_trades=True)

# Build 1-hour OHLC bars from cached daily files
bars = download_ohlc_timeseries("BTCUSDT", 7, "1h")
```

## Data Notes

- Raw daily files are downloaded from Binance Vision.
- Cached Parquet files are stored in `cache/` to avoid repeated downloads.
- The repository currently includes sample BTCUSDT daily trade files under `data/`.

## Status

This is an ongoing course project. The code and notebook will continue to evolve as new features, indicators, and strategy experiments are added.
