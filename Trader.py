from collections import deque
from typing import Dict, List

import numpy as np
import pandas as pd
from numpy import ndarray

from datamodel import OrderDepth, TradingState, Order


class Trader:
    def __init__(self) -> None:

        period_pearls = 10
        period_bananas = 10

        self.position_limits = {
            "PEARLS": 20,
            "BANANAS": 20
        }

        self.price_history = {
            "PEARLS": deque(maxlen=period_pearls),
            "BANANAS": deque(maxlen=period_bananas)
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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        for product in state.listings.keys():
            if product == 'PEARLS':
                # Implement market making strategy for PEARLS
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                acceptable_price = 10000
                spread = 0
                if len(order_depth.sell_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    print()
                    print("ASK", best_ask, best_ask_volume)
                    print()
                    if best_ask < (acceptable_price - spread):
                        print()
                        print("BUY", str(-best_ask_volume) + "x", best_ask)
                        print()
                        while best_ask_volume < -20:
                            print("BUY ASK_VOLUME", best_ask_volume)
                            orders.append(Order(product, best_ask, 20))
                            best_ask_volume += 20
                        if best_ask_volume < 0:
                            orders.append(Order(product, best_ask, -best_ask_volume))
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    print()
                    print("BID", best_bid, best_bid_volume)
                    print()
                    if best_bid > (acceptable_price + spread):
                        print()
                        print("SELL", str(best_bid_volume) + "x", best_bid)
                        print()
                        while best_bid_volume > 20:
                            print("SELL BID_VOLUME", best_bid_volume)
                            orders.append(Order(product, best_bid, -20))
                            best_bid_volume -= 20
                        if best_bid_volume > 0:
                            orders.append(Order(product, best_bid, -best_bid_volume))
                result[product] = orders
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

                            print("BUY", str(-best_ask_volume) + "x", best_ask)
                            while best_ask_volume < -20:
                                print("BUY ASK_VOLUME", best_ask_volume)
                                orders.append(Order(product, best_ask, 20))
                                best_ask_volume += 20
                            if best_ask_volume < 0:
                                orders.append(Order(product, best_ask, -best_ask_volume))
                        elif market_state == "overbought":
                            print("Not buying because market is 'overbought' ")

                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_volume = order_depth.buy_orders[best_bid]
                        if best_bid > acceptable_price and market_state != "oversold":

                            print("SELL", str(best_bid_volume) + "x", best_bid)
                            while best_bid_volume > 20:
                                print("SELL BID_VOLUME", best_bid_volume)
                                orders.append(Order(product, best_bid, -20))
                                best_bid_volume -= 20
                            if best_bid_volume > 0:
                                orders.append(Order(product, best_bid, -best_bid_volume))
                        elif market_state == "oversold":
                            print("Not selling because market is 'oversold' ")
                    result[product] = orders
        return result
