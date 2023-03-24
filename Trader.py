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
        period_coconuts = 50
        period_pinacolada = 50

        self.position_limits = {
            "PEARLS": 20,
            "BANANAS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300,
        }

        self.price_history = {
            "PEARLS": deque(maxlen=period_pearls),
            "BANANAS": deque(maxlen=period_bananas),
            "COCONUTS": deque(maxlen=period_coconuts),
            "PINA_COLADAS": deque(maxlen=period_pinacolada),
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

        orders_coco: list[Order] = []
        orders_pina: list[Order] = []

        orders: list[Order] = []

        for product in state.listings.keys():

            order_depth = state.order_depths[product]

            # STRATEGY: Market-making (stable price)
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

                            # while best_ask_volume < -20:
                            #     print("BUY ASK_VOLUME", best_ask_volume)
                            #     orders.append(Order(product, best_ask, 20))
                            #     best_ask_volume += 20
                            # if best_ask_volume < 0:
                            #     orders.append(Order(product, best_ask, -best_ask_volume))

                            if best_ask_volume < -20:
                                orders.append(Order(product, best_ask, 20))
                            else:
                                orders.append(Order(product, best_ask, -best_ask_volume))

                        elif market_state == "overbought":
                            print("Not buying because market is 'overbought' ")

                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_volume = order_depth.buy_orders[best_bid]
                        if best_bid > acceptable_price and market_state != "oversold":

                            print("SELL", str(best_bid_volume) + "x", best_bid)

                            # while best_bid_volume > 20:
                            #     print("SELL BID_VOLUME", best_bid_volume)
                            #     orders.append(Order(product, best_bid, -20))
                            #     best_bid_volume -= 20
                            # if best_bid_volume > 0:
                            #     orders.append(Order(product, best_bid, -best_bid_volume))

                            if best_bid_volume > 20:
                                orders.append(Order(product, best_bid, -20))
                            else:
                                orders.append(Order(product, best_bid, -best_bid_volume))

                        elif market_state == "oversold":
                            print("Not selling because market is 'oversold' ")
                    result[product] = orders

            # STRATEGY: MACD
            elif product in ['COCONUTS', 'PINA_COLADAS']:

                pina_order_depth = state.order_depths['PINA_COLADAS']
                coconut_order_depth = state.order_depths['COCONUTS']

                self.price_history['COCONUTS'].append(self.get_medium_price(coconut_order_depth))
                self.price_history['PINA_COLADAS'].append(self.get_medium_price(pina_order_depth))

                if len(self.price_history[product]):

                    # check correlation (just for debugging)
                    correlation_coco = self.get_correlation(self.price_history["PINA_COLADAS"],
                                                            self.price_history["COCONUTS"])
                    print(f"Correlation: {correlation_coco}")

                    pina_df = pd.DataFrame(list(self.price_history['PINA_COLADAS']))
                    coconut_df = pd.DataFrame(list(self.price_history['COCONUTS']))
                    pina_ema12 = pina_df.ewm(span=12).mean()
                    pina_ema26 = pina_df.ewm(span=26).mean()
                    pina_macd = pina_ema12 - pina_ema26
                    pina_signal = pina_macd.ewm(span=9).mean()

                    coconut_ema12 = coconut_df.ewm(span=12).mean()
                    coconut_ema26 = coconut_df.ewm(span=26).mean()
                    coconut_macd = coconut_ema12 - coconut_ema26
                    coconut_signal = coconut_macd.ewm(span=9).mean()

                    # pina coladas buy --> coconut sell
                    if pina_macd[0].values[-1] > pina_signal[0].values[-1] and coconut_macd[0].values[-1] < \
                            coconut_signal[0].values[-1]:

                        # buy pina coladas
                        if len(pina_order_depth.sell_orders) > 0:
                            pina_position = state.position["PINA_COLADAS"] if "PINA_COLADAS" in state.position else 0
                            pina_price = min(pina_order_depth.sell_orders.keys())

                            if pina_position <= 0:
                                pina_quantity = min( pina_order_depth.sell_orders[pina_price], -(300 + pina_position))
                            else:
                                pina_quantity = min( pina_order_depth.sell_orders[pina_price], -(300 - pina_position))

                            print("Buy PINA_COLADAS")
                            orders_pina.append(Order('PINA_COLADAS', pina_price, +pina_quantity))

                        # sell coconuts
                        if len(coconut_order_depth.buy_orders) > 0:
                            coconut_position = state.position["COCONUTS"] if "COCONUTS" in state.position else 0
                            coconut_price = min(coconut_order_depth.buy_orders.keys())

                            if coconut_position <= 0:
                                coconut_quantity = min(coconut_order_depth.buy_orders[coconut_price], 600 + coconut_position)
                            else:
                                coconut_quantity = min(coconut_order_depth.buy_orders[coconut_price], 600 - coconut_position)

                            print("Sell COCONUTS")
                            orders_coco.append(Order('COCONUTS', coconut_price, -coconut_quantity))

                    # pina coladas sell --> coconut buy
                    elif pina_macd[0].values[-1] < pina_signal[0].values[-1] and coconut_macd[0].values[-1] > \
                            coconut_signal[0].values[-1]:

                        # buy coconuts
                        if len(coconut_order_depth.sell_orders) > 0:
                            coconut_position = state.position["COCONUTS"] if "COCONUTS" in state.position else 0
                            coconut_price = min(coconut_order_depth.sell_orders.keys())

                            if coconut_position <= 0:
                                coconut_quantity = min(coconut_order_depth.sell_orders[coconut_price], -(300 + coconut_position))
                            else:
                                coconut_quantity = min(coconut_order_depth.sell_orders[coconut_price], -(300 - coconut_position))

                            print("Buy COCONUTS")
                            orders_coco.append(Order('COCONUTS', coconut_price, +coconut_quantity))

                        # sell pina colada
                        if len(pina_order_depth.buy_orders) > 0:
                            pina_position = state.position["PINA_COLADAS"] if "PINA_COLADAS" in state.position else 0
                            pina_price = min(pina_order_depth.buy_orders.keys())

                            if pina_position <= 0:
                                pina_quantity = min(pina_order_depth.buy_orders[pina_price], 600 + pina_position)
                            else:
                                pina_quantity = min(pina_order_depth.buy_orders[pina_price], 600 - pina_position)

                            print("Sell PINA_COLADAS")
                            orders_pina.append(Order('PINA_COLADAS', pina_price, -pina_quantity))

                # add orders
                result["PINA_COLADAS"] = orders_pina
                result["COCONUTS"] = orders_coco

        return result

