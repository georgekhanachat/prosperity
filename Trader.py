import json
from collections import deque
from math import isnan

import numpy as np
import pandas as pd
from numpy import ndarray

from datamodel import Order, ProsperityEncoder, Symbol, TradingState, OrderDepth, Product
from typing import Any, Dict, List


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        print(json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

        self.logs = ""


logger = Logger()


class Trader:

    def __init__(self):
        self.dolphin_sightings_history = deque(maxlen=20)
        self.price_history = {
            'DIVING_GEAR': deque(maxlen=20),
            'BANANAS': deque(maxlen=20),
            'COCONUTS': deque(maxlen=10),
            'PINA_COLADAS': deque(maxlen=10)
        }
        self.position_limits = {
            "PEARLS": 20,
            "BANANAS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300,
            "DIVING_GEAR": 50,
        }

    def build_macd_data_frame(self, product: Product):
        df = pd.DataFrame({"price": self.price_history[product]})
        df["EMA_12"] = df["price"].ewm(span=12).mean()
        df["EMA_26"] = df["price"].ewm(span=26).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        return df

    @staticmethod
    def get_medium_price(order_depth: OrderDepth) -> ndarray:
        """ Computes the medium price based on buy and sell orders """

        buy_orders = [float(item) for item in order_depth.buy_orders]
        sell_orders = [float(item) for item in order_depth.sell_orders]
        prices = np.array([buy_orders + sell_orders])

        return np.mean(prices)

    @staticmethod
    def get_market_state(price_history: np.ndarray) -> str:
        """ Uses Stochastic Oscillator to determine if overbought or oversold """

        # get highest and lowest values in predefined period
        high_roll = max(price_history)
        low_roll = min(price_history)

        # Fast stochastic indicator
        num = price_history - low_roll
        denom = high_roll - low_roll
        k_line = (num / denom) * 100

        # Slow stochastic indicator
        d_line = pd.DataFrame(k_line).rolling(3).mean().values

        # decide state
        if d_line[-1] > 85:
            return "overbought"
        elif d_line[-1] < 15:
            return "oversold"
        else:
            return "neutral"

    def EMA(self, periods, product: Product):
        weights = np.exp(np.linspace(-1., 0., periods))
        weights /= weights.sum()
        ema = np.convolve(self.price_history[product], weights, mode='full')[:len(self.price_history[product])]
        ema[:periods] = ema[periods]
        return ema

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        dolphin_sightings = state.observations.get("DOLPHIN_SIGHTINGS")
        if dolphin_sightings is not None:
            self.dolphin_sightings_history.append(dolphin_sightings)

        for product in state.order_depths.keys():
            if product == "DIVING_GEAR":
                order_dg = []
                order_depth: OrderDepth = state.order_depths[product]
                self.price_history[product].append(self.get_medium_price(order_depth))

                df = pd.DataFrame({
                    product: self.price_history[product],
                    "DOLPHIN_SIGHTINGS": self.dolphin_sightings_history
                })
                corr = df.corr()[product]["DOLPHIN_SIGHTINGS"]
                logger.print(corr)
                if not isnan(corr):
                    if corr > 0.6 and state.position.get(product, 0) < 50:
                        # Buy more DIVING_GEAR
                        dg_price = min(state.order_depths[product].sell_orders.keys())
                        quantity = min(50 - state.position.get(product, 0),
                                       state.order_depths[product].sell_orders[dg_price])
                        if (-quantity) + state.position[product] > 50:
                            quantity = state.position[product] - 50
                        logger.print("BUY DIVING_GEAR", dg_price, -quantity)
                        order = Order(product, dg_price, -quantity)
                        order_dg.append(order)
                    elif corr < -0.6:
                        # Sell DIVING_GEAR
                        dg_price = min(state.order_depths[product].buy_orders.keys())
                        quantity = min(state.position.get(product, 0),
                                       -state.order_depths[product].buy_orders[dg_price])
                        logger.print("SELL DIVING_GEAR", dg_price, quantity)
                        order = Order(product, dg_price, quantity)
                        order_dg.append(order)
                result[product] = order_dg
            elif product == 'PEARLS':
                # Implement market making strategy for PEARLS
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                acceptable_price = 10000
                spread = 0
                if len(order_depth.sell_orders) > 0 and state.position.get(product, 0) < 20:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = min(20 - state.position.get(product, 0), order_depth.sell_orders[best_ask])
                    if best_ask < acceptable_price - spread:
                        if -best_ask_volume + state.position.get(product, 0) > 20:
                            best_ask_volume = state.position[product] - 20
                        logger.print("BUY PEARLS", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))
                if len(order_depth.buy_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = min(state.position.get(product, 0), order_depth.buy_orders[best_bid])
                    if best_bid > acceptable_price + spread:
                        logger.print("SELL PEARLS", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))
                result[product] = orders
            elif product == "BANANAS":
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                self.price_history[product].append(self.get_medium_price(order_depth))

                # get market state and acceptable price
                acceptable_price = self.get_medium_price(order_depth)
                market_state = self.get_market_state(self.price_history[product])

                # wait for full price history (period)
                if len(self.price_history[product]):

                    if len(order_depth.sell_orders) > 0 and state.position.get(product, 0) < 20:
                        best_ask = min(order_depth.sell_orders.keys())
                        best_ask_volume = min(20 - state.position.get(product, 0), order_depth.sell_orders[best_ask])
                        if best_ask < acceptable_price and market_state != "overbought":
                            logger.print("BUY", str(-best_ask_volume) + "x", best_ask)
                            orders.append(Order(product, best_ask, -best_ask_volume))
                        elif market_state == "overbought":
                            logger.print("Not buying because market is 'overbought' ")

                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_volume = min(state.position.get(product, 0), order_depth.buy_orders[best_bid])
                        if best_bid > acceptable_price and market_state != "oversold":
                            logger.print("SELL", str(best_bid_volume) + "x", best_bid)
                            orders.append(Order(product, best_bid, -best_bid_volume))
                        elif market_state == "oversold":
                            logger.print("Not selling because market is 'oversold' ")
                result[product] = orders
            elif product in ['COCONUTS', 'PINA_COLADAS']:
                order_pc = []
                order_c = []
                pina_order_depth = state.order_depths['PINA_COLADAS']
                coconut_order_depth = state.order_depths['COCONUTS']
                self.price_history['COCONUTS'].append(self.get_medium_price(coconut_order_depth))
                self.price_history['PINA_COLADAS'].append(self.get_medium_price(pina_order_depth))

                if len(self.price_history['COCONUTS']) and len(self.price_history['PINA_COLADAS']):
                    df = pd.DataFrame(
                        {"PINA_COLADAS": self.price_history["PINA_COLADAS"], "COCONUTS": self.price_history["COCONUTS"]})
                    corr = df.corr()["PINA_COLADAS"]["COCONUTS"]

                    logger.print(corr)
                    if not isnan(corr):
                        if corr > 0.6 and state.position.get("PINA_COLADAS", 0) < self.position_limits["PINA_COLADAS"]:
                            # Buy more PINA_COLADAS
                            pc_price = min(pina_order_depth.sell_orders.keys())
                            quantity = min(self.position_limits["PINA_COLADAS"] - state.position.get("PINA_COLADAS", 0),
                                           pina_order_depth.sell_orders[pc_price])
                            if (-quantity) + state.position.get("PINA_COLADAS", 0) > self.position_limits["PINA_COLADAS"]:
                                quantity = state.position.get("PINA_COLADAS", 0) - self.position_limits["PINA_COLADAS"]
                            logger.print("BUY PINA_COLADAS", pc_price, -quantity)
                            order = Order("PINA_COLADAS", pc_price, -quantity)
                            order_pc.append(order)
                        elif corr < -0.6:
                            # Sell PINA_COLADAS
                            pc_price = min(pina_order_depth.buy_orders.keys())
                            quantity = min(state.position.get("PINA_COLADAS", 0),
                                           -pina_order_depth.buy_orders[pc_price])
                            logger.print("SELL PINA_COLADAS", pc_price, quantity)
                            order = Order("PINA_COLADAS", pc_price, quantity)
                            order_pc.append(order)

                result['PINA_COLADAS'] = order_pc

        logger.flush(state, result)
        return result
