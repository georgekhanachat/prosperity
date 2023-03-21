from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
from collections import deque


class Trader:
    def __init__(self, period: int = 10):
        self.period = period
        self.banana_ma = deque(maxlen=period)

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        for product in state.order_depths.keys():
            if product == 'PEARLS':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                acceptable_price = 10000

                if len(order_depth.sell_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]

                    if best_ask < acceptable_price:
                        print("BUY", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]

                    if best_bid > acceptable_price:
                        print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

                result[product] = orders

            if product == 'BANANAS':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                banana_price = list(order_depth.sell_orders.keys())[0]
                self.banana_ma.append(banana_price)

                if len(self.banana_ma) == self.period:
                    if self.banana_ma[-1] < sum(self.banana_ma) / self.period:
                        if state.position[product] < 20:
                            quantity = 20 - state.position[product]
                            print("BUY", str(quantity) + "x", banana_price)
                            orders.append(Order(product, banana_price, quantity))

                    if self.banana_ma[-1] > sum(self.banana_ma) / self.period:
                        if state.position[product] > -20:
                            quantity = -20 - state.position[product]
                            print("SELL", str(-quantity) + "x", banana_price)
                            orders.append(Order(product, banana_price, -quantity))

                result[product] = orders

        return result
