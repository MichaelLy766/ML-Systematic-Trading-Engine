"""Minimal live Binance futures execution helpers.

This module turns the notebook's `Order` objects into authenticated Binance
USD-M futures REST calls. It does not manage strategy state or price feeds;
it only handles exchange connectivity, position lookup, and market orders.
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
import hashlib
import hmac
import os
import threading
import time
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode

import requests
import websocket


@dataclass(frozen=True)
class BinanceOrderResult:
    symbol: str
    side: str
    quantity: Decimal
    raw: Dict[str, Any]


class BinanceFuturesExchange:
    """Small REST client for Binance USD-M futures."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        region: str = "global",
        recv_window: int = 5000,
    ) -> None:
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET are required")

        if region == "us":
            # Binance.US does not have a testnet; always use production
            self.base_url = "https://fapi.binanceus.com"
        else:
            # Global Binance (testnet or production)
            self.base_url = "https://demo-fapi.binance.com" if testnet else "https://fapi.binance.com"
        
        self.recv_window = recv_window
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})
        self._exchange_info: Optional[Dict[str, Any]] = None

    def _public_request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def _signed_request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(params or {})
        params["timestamp"] = self._timestamp_ms()
        params["recvWindow"] = self.recv_window

        query_string = urlencode(params, doseq=True)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature

        response = self.session.request(
            method=method,
            url=f"{self.base_url}{path}",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _timestamp_ms() -> int:
        from time import time

        return int(time() * 1000)

    def exchange_info(self) -> Dict[str, Any]:
        if self._exchange_info is None:
            self._exchange_info = self._public_request("/fapi/v1/exchangeInfo")
        return self._exchange_info

    def _symbol_filters(self, symbol: str) -> Dict[str, Any]:
        for item in self.exchange_info()["symbols"]:
            if item["symbol"] == symbol:
                return {f["filterType"]: f for f in item["filters"]}
        raise ValueError(f"Unknown Binance futures symbol: {symbol}")

    @staticmethod
    def _round_down_to_step(quantity: Decimal, step_size: Decimal) -> Decimal:
        if step_size <= 0:
            return quantity
        steps = (quantity / step_size).to_integral_value(rounding=ROUND_DOWN)
        return steps * step_size

    def normalized_quantity(self, symbol: str, quantity: Decimal) -> Decimal:
        filters = self._symbol_filters(symbol)
        lot_size = filters.get("LOT_SIZE")
        if lot_size is None:
            return quantity

        step_size = Decimal(lot_size["stepSize"])
        min_qty = Decimal(lot_size["minQty"])
        rounded = self._round_down_to_step(quantity, step_size)
        if rounded < min_qty:
            raise ValueError(f"Quantity {quantity} is below Binance minimum {min_qty} for {symbol}")
        return rounded

    def get_mark_price(self, symbol: str) -> Decimal:
        data = self._public_request("/fapi/v1/ticker/price", {"symbol": symbol})
        return Decimal(str(data["price"]))

    def get_usdt_balance(self) -> Decimal:
        balances = self._signed_request("GET", "/fapi/v2/balance")
        for item in balances:
            if item["asset"] == "USDT":
                return Decimal(str(item["balance"]))
        raise ValueError("USDT balance not found")

    def balance(self) -> Decimal:
        return self.get_usdt_balance()

    def get_position(self, symbol: str) -> Decimal:
        positions = self._signed_request("GET", "/fapi/v2/positionRisk")
        for item in positions:
            if item["symbol"] == symbol:
                return Decimal(str(item["positionAmt"]))
        return Decimal("0")

    def market_order(self, symbol: str, signed_qty: Decimal) -> BinanceOrderResult:
        quantity = Decimal(str(abs(signed_qty)))
        if quantity == 0:
            return BinanceOrderResult(symbol=symbol, side="NONE", quantity=Decimal("0"), raw={"status": "SKIPPED"})

        side = "BUY" if signed_qty > 0 else "SELL"
        quantity = self.normalized_quantity(symbol, quantity)

        payload = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": str(quantity),
            "newOrderRespType": "RESULT",
        }
        data = self._signed_request("POST", "/fapi/v1/order", payload)
        return BinanceOrderResult(symbol=symbol, side=side, quantity=quantity, raw=data)


class BinanceMarkPriceWebsocket:
    """Stream Binance USD-M futures mark prices for one symbol."""

    def __init__(
        self,
        symbol: str,
        on_price: Callable[[Decimal, Dict[str, Any]], None],
        testnet: bool = True,
        region: str = "global",
        reconnect_delay: float = 5.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.on_price = on_price
        self.testnet = testnet
        self.reconnect_delay = reconnect_delay
        self.latest_price: Optional[Decimal] = None
        self.latest_payload: Optional[Dict[str, Any]] = None
        self._stop_event = threading.Event()
        self._stream_name = f"{self.symbol.lower()}@markPrice@1s"
        if region == "us":
            base_ws = "wss://stream.binanceus.com/ws"
        else:
            base_ws = "wss://stream.binancefuture.com/ws" if testnet else "wss://fstream.binance.com/ws"
        self._url = f"{base_ws}/{self._stream_name}"
        self._app = websocket.WebSocketApp(
            self._url,
            on_message=self._handle_message,
            on_error=self._handle_error,
            on_close=self._handle_close,
        )

    def _handle_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        payload = json.loads(message)
        price = Decimal(str(payload["p"]))
        self.latest_price = price
        self.latest_payload = payload
        self.on_price(price, payload)

    def _handle_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        if not self._stop_event.is_set():
            print(f"[websocket error] {error}")

    def _handle_close(self, _ws: websocket.WebSocketApp, close_status_code: int, close_msg: str) -> None:
        if not self._stop_event.is_set():
            print(f"[websocket closed] {close_status_code} {close_msg}")

    def stop(self) -> None:
        self._stop_event.set()
        self._app.close()

    def run_forever(self) -> None:
        while not self._stop_event.is_set():
            self._app.run_forever(ping_interval=30, ping_timeout=10)
            if not self._stop_event.is_set():
                time.sleep(self.reconnect_delay)


def run_live_strategy_loop(
    strategy: Any,
    exchange: BinanceFuturesExchange,
    symbol: str,
    testnet: bool = True,
    region: str = "global",
) -> BinanceMarkPriceWebsocket:
    """Run the strategy on a Binance mark-price websocket and execute orders live.

    The exchange object is passed into the strategy because it exposes the
    `balance()` and `get_position()` methods the notebook strategy expects.
    """

    def on_price(price: Decimal, _payload: Dict[str, Any]) -> None:
        try:
            orders = strategy.on_tick(float(price), exchange)
            if orders:
                for order in orders:
                    qty = order.signed_qty
                    if qty != Decimal("0"):
                        # Binance handles shorting properly, just execute
                        result = exchange.market_order(symbol, qty)
                        
                        # Log to CSV
                        log_file = "stats/trading_log.csv"
                        os.makedirs("stats", exist_ok=True)
                        file_exists = os.path.exists(log_file)
                        with open(log_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["timestamp", "symbol", "side", "quantity", "price", "balance", "position", "prediction"])
                            
                            writer.writerow([
                                datetime.utcnow().isoformat(),
                                symbol,
                                result.side,
                                str(result.quantity),
                                str(price),
                                str(exchange.balance()),
                                str(exchange.get_position(symbol)),
                                f"{order.prediction:.6f}"
                            ])
        except Exception as e:
            print(f"Error in strategy tick: {e}")

    stream = BinanceMarkPriceWebsocket(symbol, on_price, testnet=testnet, region=region)
    t = threading.Thread(target=stream.run_forever, daemon=True)
    t.start()
    return stream
