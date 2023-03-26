from collections import deque
from typing import Dict, List

import numpy as np
import pandas as pd
from numpy import ndarray
from math import isnan

from datamodel import Order, ProsperityEncoder, Symbol, TradingState, OrderDepth
from typing import Any, Dict, List
import json

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
    def __init__(self) -> None:

        period_pearls = 10
        period_bananas = 10
        period_coconuts = 50
        period_pinacolada = 50
        period_diving_gear = 20
        
        self.dolphin_sightings_history = deque(maxlen=period_diving_gear)

        self.position_limits = {
            "PEARLS": 20,
            "BANANAS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300,
            "DIVING_GEAR": 50,
        }

        self.price_history = {
            "PEARLS": deque(maxlen=period_pearls),
            "BANANAS": deque(maxlen=period_bananas),
            "COCONUTS": deque(maxlen=period_coconuts),
            "PINA_COLADAS": deque(maxlen=period_pinacolada),
            'DIVING_GEAR': deque(maxlen=period_diving_gear)
        }

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

    @staticmethod
    def get_medium_price(order_depth: OrderDepth) -> ndarray:
        """ Computes the medium price based on buy and sell orders """

        buy_orders = [float(item) for item in order_depth.buy_orders]
        sell_orders = [float(item) for item in order_depth.sell_orders]
        prices = np.array([buy_orders + sell_orders])

        return np.mean(prices)

    @staticmethod
    def get_correlation(a: List, b: List) -> float:
        """ Returns pearson correlation of a to b """

        pearson_corr = np.corrcoef(a, b)
        return pearson_corr[0][1]

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        order_dg: list[Order] = []
        dolphin_sightings = state.observations.get("DOLPHIN_SIGHTINGS")
        if dolphin_sightings is not None:
            self.dolphin_sightings_history.append(dolphin_sightings)

        orders: list[Order] = []
        for product in state.listings.keys():

            # skip non tradable products
            if product == "DOLPHIN_SIGHTINGS":
                continue

            # get products order depth
            order_depth = state.order_depths[product]

            # STRATEGY: Market-making (stable price)
            if product == 'PEARLS':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                acceptable_price = 10000
                spread = 0
                if len(order_depth.sell_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                
                    if best_ask < (acceptable_price - spread):
                        
                        while best_ask_volume < -20:
                            orders.append(Order(product, best_ask, 20))
                            best_ask_volume += 20
                        if best_ask_volume < 0:
                            orders.append(Order(product, best_ask, -best_ask_volume))
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid > (acceptable_price + spread):
                        while best_bid_volume > 20:
                            orders.append(Order(product, best_bid, -20))
                            best_bid_volume -= 20
                        if best_bid_volume > 0:
                            orders.append(Order(product, best_bid, -best_bid_volume))
                result[product] = orders
                
            # STRATEGY: Stochastic Oscillators (NOTE: not working well)
            elif product == 'BANANAS':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                self.price_history[product].append(self.get_medium_price(order_depth))

                # get market state and acceptable price
                acceptable_price = self.get_medium_price(order_depth)
                market_state = self.get_market_state(self.price_history[product])

                # wait for full price history (period)
                if len(self.price_history[product]):

                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        best_ask_volume = order_depth.sell_orders[best_ask]
                        if best_ask < acceptable_price and market_state != "overbought":

                            if best_ask_volume < -20:
                                orders.append(Order(product, best_ask, 20))
                            else:
                                orders.append(Order(product, best_ask, -best_ask_volume))

                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_volume = order_depth.buy_orders[best_bid]
                        if best_bid > acceptable_price and market_state != "oversold":

                            if best_bid_volume > 20:
                                orders.append(Order(product, best_bid, -20))
                            else:
                                orders.append(Order(product, best_bid, -best_bid_volume))

                    result[product] = orders
            
            elif product == "COCONUTS":
                # TODO: strategy on coconuts
                continue        
            
            elif product == "PINA_COLADAS":
                # NOTE: use same strategy on COCONUTS/PINA_COLADAS as for DIVING_GEAR 
                
                pina_order_depth = state.order_depths['PINA_COLADAS']
                coconut_order_depth = state.order_depths['COCONUTS']
                self.price_history['COCONUTS'].append(self.get_medium_price(coconut_order_depth))
                self.price_history['PINA_COLADAS'].append(self.get_medium_price(pina_order_depth))
                position_coco = state.position[product] if product in state.position else 0
                
                df = pd.DataFrame({"PINA_COLADAS": self.price_history["PINA_COLADAS"], "COCONUTS": self.price_history[product]})
                corr = df.corr()["PINA_COLADAS"]["COCONUTS"]
                logger.print(corr)
                if not isnan(corr):
                    if corr > 0.6 and state.position.get("PINA_COLADAS", 0) < self.position_limits["PINA_COLADAS"]:
                        # Buy more PINA_COLADAS
                        dg_price = min(state.order_depths["PINA_COLADAS"].sell_orders.keys())
                        quantity = min(self.position_limits["PINA_COLADAS"] - state.position.get("PINA_COLADAS", 0),
                                    state.order_depths["PINA_COLADAS"].sell_orders[dg_price])
                        if position_coco > 0 and (-quantity) + position_coco > self.position_limits["PINA_COLADAS"]:
                            quantity = position_coco - self.position_limits["PINA_COLADAS"]
                        logger.print("BUY PINA_COLADAS", dg_price, -quantity)
                        order = Order("PINA_COLADAS", dg_price, -quantity)
                        order_dg.append(order)
                    elif corr < -0.6:
                        # Sell PINA_COLADAS
                        dg_price = min(state.order_depths["PINA_COLADAS"].buy_orders.keys())
                        quantity = min(state.position.get("PINA_COLADAS", 0),
                                    -state.order_depths["PINA_COLADAS"].buy_orders[dg_price])
                        logger.print("SELL PINA_COLADAS", dg_price, quantity)
                        order = Order("PINA_COLADAS", dg_price, quantity)
                        order_dg.append(order)
                result['COCONUTS'] = order_dg
                
            if product == "DIVING_GEAR":
                self.price_history[product].append(self.get_medium_price(order_depth))

                df = pd.DataFrame({"DIVING_GEAR": self.price_history[product], "DOLPHIN_SIGHTINGS": self.dolphin_sightings_history})
                corr = df.corr()["DIVING_GEAR"]["DOLPHIN_SIGHTINGS"]
                logger.print(corr)
                if not np.isnan(corr):
                    if corr > 0.6 and state.position.get("DIVING_GEAR", 0) < 50:
                        # Buy more DIVING_GEAR
                        dg_price = min(state.order_depths["DIVING_GEAR"].sell_orders.keys())
                        quantity = min(50 - state.position.get("DIVING_GEAR", 0),
                                    state.order_depths["DIVING_GEAR"].sell_orders[dg_price])
                        if state.position[product] > 0 and (-quantity) + state.position[product] > 50:
                            quantity = state.position[product] - 50
                        logger.print("BUY DIVING_GEAR", dg_price, -quantity)
                        order = Order("DIVING_GEAR", dg_price, -quantity)
                        order_dg.append(order)
                    elif corr < -0.6:
                        # Sell DIVING_GEAR
                        dg_price = min(state.order_depths["DIVING_GEAR"].buy_orders.keys())
                        quantity = min(state.position.get("DIVING_GEAR", 0),
                                    -state.order_depths["DIVING_GEAR"].buy_orders[dg_price])
                        logger.print("SELL DIVING_GEAR", dg_price, quantity)
                        order = Order("DIVING_GEAR", dg_price, quantity)
                        order_dg.append(order)
                result['DIVING_GEAR'] = order_dg
                
        logger.flush(state, result)
        return result

