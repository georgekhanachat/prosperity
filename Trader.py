import json
from collections import deque
from math import isnan

import numpy as np
import pandas as pd
from numpy import ndarray

from datamodel import Order, ProsperityEncoder, Symbol, TradingState, OrderDepth
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
            'DIVING_GEAR': deque(maxlen=20)
        }

    # @staticmethod
    # def calculate_correlation(x: list, y: list) -> float:
    #     """Calculate the correlation coefficient between two arrays."""
    #     # Calculate the mean of each array
    #     x_mean = np.mean(x)
    #     y_mean = np.mean(y)
    #
    #     # Calculate the standard deviation of each array
    #     x_std = np.std(x)
    #     y_std = np.std(y)
    #
    #     # Calculate the covariance between the two arrays
    #     covariance = np.sum((x - x_mean) * (y - y_mean)) / len(x)
    #
    #     # Calculate the correlation coefficient
    #     correlation = covariance / (x_std * y_std)
    #
    #     return correlation

    @staticmethod
    def get_medium_price(order_depth: OrderDepth) -> ndarray:
        """ Computes the medium price based on buy and sell orders """

        buy_orders = [float(item) for item in order_depth.buy_orders]
        sell_orders = [float(item) for item in order_depth.sell_orders]
        prices = np.array([buy_orders + sell_orders])

        return np.mean(prices)

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

                df = pd.DataFrame(
                    {"DIVING_GEAR": self.price_history[product], "DOLPHIN_SIGHTINGS": self.dolphin_sightings_history})
                corr = df.corr()["DIVING_GEAR"]["DOLPHIN_SIGHTINGS"]
                logger.print(corr)
                if not isnan(corr):
                    if corr > 0.6 and state.position.get("DIVING_GEAR", 0) < 50:
                        # Buy more DIVING_GEAR
                        dg_price = min(state.order_depths["DIVING_GEAR"].sell_orders.keys())
                        quantity = min(50 - state.position.get("DIVING_GEAR", 0),
                                       state.order_depths["DIVING_GEAR"].sell_orders[dg_price])
                        if (-quantity) + state.position[product] > 50:
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
