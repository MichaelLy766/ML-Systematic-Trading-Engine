import time
import os
import threading
from collections import deque
from typing import Generic, TypeVar, Optional, List, Any, Dict
from dataclasses import dataclass
from decimal import Decimal

import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv

import models
from live_exchange import BinanceFuturesExchange, run_live_strategy_loop

T = TypeVar('T')
R = TypeVar('R')

class Tick(Generic[T, R]):
    def on_tick(self, val: T) -> R:
        pass

class IntervalFeatureExtractor:
    def __init__(self, symbol: str, interval: str, num_lags: int):
        self.symbol = symbol.replace('/', '') # Binance expects BTCUSDT
        self.interval = interval
        self.num_lags = num_lags
        self.historical_returns = []
        self.last_close_price = None

    def _fetch_historical(self):
        import requests
        import math
        # Fetch completed candles from Binance public API
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={self.symbol}&interval={self.interval}&limit={self.num_lags + 2}"
        data = requests.get(url).json()
        
        # Binance kline: [open_time, open, high, low, close, ...]
        closes = [float(candle[4]) for candle in data]
        
        returns = []
        for i in range(1, len(closes)):
            returns.append(math.log(closes[i] / closes[i-1]))
            
        # The second-to-last item is the last fully COMPLETED candle.
        # The very last item is the CURRENT, IN-PROGRESS candle.
        self.last_close_price = float(data[-2][4])
        self.historical_returns = returns[:-1] 

    def get_features(self, live_price: float) -> Optional[torch.Tensor]:
        import math
        if not self.historical_returns:
            self._fetch_historical()
            
        current_return = math.log(live_price / self.last_close_price)
        
        if self.num_lags == 1:
            features = [current_return]
        else:
            features = self.historical_returns[-(self.num_lags - 1):] + [current_return]
            
        return torch.tensor([features], dtype=torch.float32)

@dataclass(frozen=True)
class Order:
    sym: str
    signed_qty: Decimal
    prediction: float = 0.0

class LiveTakerStrat:
    def __init__(self, sym: str, model: nn.Module, feature_extractor: IntervalFeatureExtractor, scale_factor: Decimal):
        self.sym = sym
        self.model = model
        self.feature_extractor = feature_extractor
        self.scale_factor = scale_factor

    def _signed_compound_trade_size(self, y_hat: float, account: Any, cur_price: Decimal) -> Decimal:
        dir_signal = np.sign(y_hat)
        # account.balance() on Alpaca is portfolio_value which already includes unrealized PnL
        cur_balance = account.balance()
        qty = cur_balance / cur_price
        signed_qty = Decimal(dir_signal) * qty
        return signed_qty * self.scale_factor

    def _create_orders(self, y_hat: float, account: Any, price: Decimal) -> List[Order]:
        pos_qty = account.get_position(self.sym)
        target_qty = self._signed_compound_trade_size(y_hat, account, price)
        
        # Only trade if the model's desired direction is different from our current position direction!
        # (Otherwise it constantly closes and re-opens the exact same position, destroying PnL via fees)
        current_dir = np.sign(float(pos_qty))
        target_dir = np.sign(y_hat)
        
        if current_dir == target_dir and pos_qty != Decimal("0"):
            return []
        
        open_order = Order(self.sym, target_qty, prediction=y_hat)
        
        # If we already have a position, close it first before opening the new one
        if pos_qty != Decimal("0"):
            close_order = Order(self.sym, -pos_qty, prediction=y_hat)
            return [close_order, open_order]
        return [open_order]      

    def on_tick(self, price: float, account: Any) -> List[Order]:
        X = self.feature_extractor.get_features(price)
        if X is not None:
            with torch.no_grad():                
                y_hat = self.model(X)
                return self._create_orders(y_hat.item(), account, Decimal(price))
        return []

def main():
    load_dotenv()
    
    # 1. Load Model
    model = models.LinearModel(1)
    model.load_state_dict(torch.load('trading_strategy/1d_lag3_model.pth', weights_only=True))
    model.eval()

    # 2. Initialize Binance Exchange Adapter
    live_exchange = BinanceFuturesExchange(testnet=True)
    try:
        print(f"Current Binance Testnet Balance: ${live_exchange.balance()}")
    except Exception as e:
        print(f"Failed to connect to Binance Testnet: {e}")
        print("Please ensure your .env file has valid BINANCE_API_KEY and BINANCE_API_SECRET")
        return

    # 3. Initialize Strategy with proper interval features
    # Adjust this interval ('1d', '12h', '1m', etc.) and num_lags to perfectly match the model you loaded above!
    extractor = IntervalFeatureExtractor(symbol='BTCUSDT', interval='1d', num_lags=1)
    
    live_strat = LiveTakerStrat(
        sym='BTCUSDT', 
        model=model, 
        feature_extractor=extractor, 
        scale_factor=Decimal('0.1')
    )

    # 4. Start the WebSocket and trade loop
    print("Starting Live Strategy Loop. Press Ctrl+C to stop.")
    ws = run_live_strategy_loop(live_strat, live_exchange, symbol="BTCUSDT", testnet=True)
    
    try:
        ticks_processed = 0
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=== Binance Live Strategy Monitor ===")
            print(f"Ticks processed:  {ticks_processed}")
            
            try:
                pos = live_exchange.get_position('BTCUSDT')
                print(f"Current Position: {pos} BTC")
            except Exception as e:
                print(f"Current Position: [Error fetching: {e}]")
                
            print(f"Latest Price:     ${ws.latest_price if ws.latest_price else 'Waiting for tick...'}")
            
            if ws.latest_payload:
                print(f"Last payload:     {ws.latest_payload}")
                ticks_processed += 1
                
            time.sleep(1.0) 
    except KeyboardInterrupt:
        print("\nLive monitor stopping...")
        ws.stop()

if __name__ == "__main__":
    main()
