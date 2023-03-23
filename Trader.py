import json
from collections import deque

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from numpy import ndarray

from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder, Symbol, Product


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

        self.position_limits = {
            "PEARLS": 20,
            "BANANAS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300,
        }

        self.remaining_pc = self.position_limits["PINA_COLADAS"]
        self.remaining_c = self.position_limits["COCONUTS"]

        self.price_history = {
            "PEARLS": deque(maxlen=period_pearls),
            "BANANAS": deque(maxlen=period_bananas),
            "COCONUTS": deque(maxlen=period_coconuts),
            "PINA_COLADAS": deque(maxlen=period_pinacolada),
        }

    def buildDataFrame(self, orderDepth: OrderDepth, product: Product):
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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        for product in state.listings.keys():
            if product in ['COCONUTS', 'PINA_COLADAS']:
                # calculate MACD for PINA_COLADAS and COCONUTS
                pc_order_depth: OrderDepth = state.order_depths["PINA_COLADAS"]
                c_order_depth: OrderDepth = state.order_depths["COCONUTS"]

                self.price_history['COCONUTS'].append(self.get_medium_price(c_order_depth))
                self.price_history['PINA_COLADAS'].append(self.get_medium_price(pc_order_depth))

                if len(self.price_history[product]):
                    pc_df = self.buildDataFrame(pc_order_depth, "PINA_COLADAS")
                    c_df = self.buildDataFrame(pc_order_depth, "COCONUTS")

                    logger.print("p", pc_df.iloc[-1]["MACD"], pc_df.iloc[-1]["Signal"])
                    logger.print("c", c_df.iloc[-1]["MACD"], c_df.iloc[-1]["Signal"])

                    # implement pair trading strategy
                    if pc_df.iloc[-1]["MACD"] > pc_df.iloc[-1]["Signal"] and c_df.iloc[-1]["MACD"] < c_df.iloc[-1][
                        "Signal"]:
                        # bullish signal, buy PINA_COLADAS and sell COCONUTS
                        # buy pina coladas
                        if len(pc_order_depth.sell_orders) > 0:
                            pina_position = state.position["PINA_COLADAS"] if "PINA_COLADAS" in state.position else 0
                            pina_price = min(pc_order_depth.sell_orders.keys())
                            pina_quantity = min(pc_order_depth.sell_orders[pina_price], 300 - pina_position)
                            logger.print("Buy PINA_COLADAS")
                            result.setdefault("PINA_COLADAS", []).append(Order('PINA_COLADAS', pina_price, pina_quantity))

                        # sell coconuts
                        if len(c_order_depth.buy_orders) > 0:
                            coconut_position = state.position["COCONUTS"] if "COCONUTS" in state.position else 0
                            coconut_price = min(c_order_depth.buy_orders.keys())
                            coconut_quantity = min(c_order_depth.buy_orders[coconut_price], 600 - coconut_position)
                            logger.print("Sell COCONUTS")
                            result.setdefault("COCONUTS", []).append(Order('COCONUTS', coconut_price, - coconut_quantity))

                    elif pc_df.iloc[-1]["MACD"] < pc_df.iloc[-1]["Signal"] and c_df.iloc[-1]["MACD"] > c_df.iloc[-1][
                        "Signal"]:
                        # bearish signal, sell PINA_COLADAS and buy COCONUTS
                        if len(c_order_depth.sell_orders) > 0:
                            coconut_position = state.position["COCONUTS"] if "COCONUTS" in state.position else 0
                            coconut_price = min(c_order_depth.sell_orders.keys())
                            coconut_quantity = min(c_order_depth.sell_orders[coconut_price], 300 - coconut_position)
                            logger.print("Buy COCONUTS")
                            result.setdefault("COCONUTS", []).append(Order('COCONUTS', coconut_price, coconut_quantity))

                            # sell coconuts
                        if len(pc_order_depth.buy_orders) > 0:
                            pina_position = state.position["PINA_COLADAS"] if "PINA_COLADAS" in state.position else 0
                            pina_price = min(pc_order_depth.buy_orders.keys())
                            pina_quantity = min(pc_order_depth.buy_orders[pina_price], 600 - pina_position)
                            logger.print("Sell PINA_COLADAS")
                            result.setdefault("PINA_COLADAS", []).append(Order('PINA_COLADAS', pina_price, - pina_quantity))
                    logger.print("result", result)
                    logger.flush(state, result)
        return result
