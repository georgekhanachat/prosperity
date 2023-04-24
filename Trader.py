import json
from collections import deque
from typing import Any
from typing import Dict, List

import numpy
import numpy as np
import pandas as pd
from numpy import ndarray

from datamodel import Order, ProsperityEncoder, Symbol, TradingState, Product
from datamodel import OrderDepth


class Logger:
    # Set this to true, if u want to create
    # local logs
    local: bool
    # this is used as a buffer for logs
    # instead of stdout
    local_logs: dict[int, str] = {}

    def __init__(self, local=False) -> None:
        self.logs = ""
        self.local = local

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        output = json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True)
        if self.local:
            self.local_logs[state.timestamp] = output
        print(output)

        self.logs = ""


logger = Logger(local=False)


class Trader:

    def __init__(self):
        self.price_history = {
            'BANANAS': deque(maxlen=10)
        }
        self.bananas_sum = 0
        self.pearls_sum = 0
        self.b_num = 0
        self.p_num = 0

        self.coco_ask = self.coco_bid = self.pina_ask = self.pina_bid = 0

        self.bd = []
        self.bl = []
        self.last_buy = 0

        self.last_berries = 1000000

    @staticmethod
    def market_making_trade(state: TradingState, product: Product, position_limit: int, acceptable_price: int,
                            spread: int):
        order_depth: OrderDepth = state.order_depths[product]
        orders: list[Order] = []
        if len(order_depth.sell_orders) > 0 and state.position.get(product, 0) < position_limit:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = min(position_limit - state.position.get(product, 0), order_depth.sell_orders[best_ask])
            if best_ask < (acceptable_price - spread):
                logger.print("BUY", str(-best_ask_volume) + "x", best_ask)
                orders.append(Order(product, best_ask, -best_ask_volume))
        if len(order_depth.buy_orders) != 0 and state.position.get(product, 0) > -position_limit:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = min(position_limit + state.position.get(product, 0), order_depth.buy_orders[best_bid])
            if best_bid > (acceptable_price + spread):
                logger.print("SELL", str(best_bid_volume) + "x", best_bid)
                orders.append(Order(product, best_bid, -best_bid_volume))
        return orders

    @staticmethod
    def get_medium_price(order_depth: OrderDepth) -> ndarray:
        """ Computes the medium price based on buy and sell orders """

        buy_orders = [float(item) for item in order_depth.buy_orders]
        sell_orders = [float(item) for item in order_depth.sell_orders]
        prices = np.array([max(buy_orders) + min(sell_orders)])

        return np.mean(prices)

    def run(self, state: TradingState) -> Dict[str, List[Order]]:

        def get_best_orders(book: OrderDepth):
            return max(book.buy_orders.keys()), min(book.sell_orders.keys())

        result = {}
        for product in state.order_depths.keys():
            if product == 'PEARLS':
                result[product] = self.market_making_trade(state, product, 20, 10000, 0)
            elif product == 'BANANAS':
                order_depth: OrderDepth = state.order_depths[product]
                self.price_history[product].append(self.get_medium_price(order_depth))
                if len(self.price_history[product]):
                    ma5 = pd.Series(list(self.price_history[product])).rolling(window=5).mean()
                    ma5 = ma5.iloc[-1]
                    result[product] = self.market_making_trade(state, product, 20, ma5, 0)

        CUTOFF = 2
        mean_ratio = 15000 / 8000

        coco_book: OrderDepth = state.order_depths["COCONUTS"]
        pina_book: OrderDepth = state.order_depths["PINA_COLADAS"]

        coco_best_ask = min(coco_book.sell_orders.keys())
        pina_best_buy = max(pina_book.buy_orders.keys())

        pina_best_ask = min(pina_book.sell_orders.keys())
        coco_best_buy = max(coco_book.buy_orders.keys())

        cutoff_1 = cutoff_2 = CUTOFF
        coco_pos = state.position.get("COCONUTS", 0)
        pina_pos = state.position.get("PINA_COLADAS", 0)
        if pina_pos > 0 and coco_pos < 0:
            cutoff_1 *= 5
        if pina_pos < 0 and coco_pos > 0:
            cutoff_2 *= 5

        diff1 = coco_best_buy * mean_ratio - pina_best_ask
        if diff1 > cutoff_1 and state.position.get('PINA_COLADAS', 0) < 300 and state.position.get('COCONUTS',
                                                                                                   0) > -600:
            pina_vol = min(300 - state.position.get('PINA_COLADAS', 0), -pina_book.sell_orders[pina_best_ask])
            coco_vol = min(600 + state.position.get('COCONUTS', 0), coco_book.buy_orders[coco_best_buy])
            trade_vol = min(pina_vol, coco_vol // 2, diff1 // cutoff_1)
            result["PINA_COLADAS"] = [Order("PINA_COLADAS", pina_best_ask, trade_vol)]
            result["COCONUTS"] = [
                Order("COCONUTS", coco_best_buy, -trade_vol * 2)]

        diff2 = pina_best_buy - coco_best_ask * mean_ratio
        if diff2 > cutoff_2 and state.position.get('PINA_COLADAS', 0) > -300 and state.position.get('COCONUTS',
                                                                                                    0) < 600:
            pina_vol = min(300 + state.position.get('PINA_COLADAS', 0), pina_book.buy_orders[pina_best_buy])
            coco_vol = min(600 - state.position.get('COCONUTS', 0), -coco_book.sell_orders[coco_best_ask])
            trade_vol = min(pina_vol, coco_vol // 2, diff2 // cutoff_2)
            result["PINA_COLADAS"] = [Order("PINA_COLADAS", pina_best_buy, -trade_vol)]
            result["COCONUTS"] = [
                Order("COCONUTS", coco_best_ask, trade_vol * 2)]

        picnic_book: OrderDepth = state.order_depths["PICNIC_BASKET"]
        dip_book: OrderDepth = state.order_depths["DIP"]
        bread_book: OrderDepth = state.order_depths["BAGUETTE"]
        uk_book: OrderDepth = state.order_depths["UKULELE"]

        result["DIP"] = []
        result["BAGUETTE"] = []
        result["UKULELE"] = []
        result["PICNIC_BASKET"] = []

        picnic_bid, picnic_ask = get_best_orders(picnic_book)

        bp = state.position.get("PICNIC_BASKET", 0) * 4
        offset = - 370 + bp // 2

        while True:
            try:
                buy_cost = 0

                dips = 0
                breads = 0

                temp_orders = {"BAGUETTE": [], "DIP": [], "UKULELE": []}

                while dips < 4:
                    dip_sell = get_best_orders(dip_book)[1]
                    qty = min(4 - dips, -dip_book.sell_orders[dip_sell])
                    dips += qty

                    del dip_book.sell_orders[dip_sell]

                    buy_cost += dip_sell * qty
                    temp_orders["DIP"].append(Order("DIP", dip_sell, qty))  # buy

                while breads < 2:
                    bread_sell = get_best_orders(bread_book)[1]
                    qty = min(2 - breads, -bread_book.sell_orders[bread_sell])
                    breads += qty

                    del bread_book.sell_orders[bread_sell]

                    buy_cost += bread_sell * qty
                    temp_orders["BAGUETTE"].append(Order("BAGUETTE", bread_sell, qty))  # buy

                uk_sell = get_best_orders(uk_book)[1]
                qty = 1

                buy_cost += uk_sell * qty
                temp_orders["UKULELE"].append(Order("UKULELE", uk_sell, qty))  # buy

                logger.print("sell basket for", picnic_bid, "buy items for", buy_cost)

                if buy_cost < picnic_bid + offset:
                    result["PICNIC_BASKET"].append(Order("PICNIC_BASKET", picnic_bid, -1))  # sell basket
                    result["DIP"].extend(temp_orders["DIP"])  # buy the rest
                    result["UKULELE"].extend(temp_orders["UKULELE"])
                    result["BAGUETTE"].extend(temp_orders["BAGUETTE"])
                else:

                    break
            except:
                break

        while True:
            try:
                sell_cost = 0

                dips = 0
                breads = 0

                temp_orders = {"BAGUETTE": [], "DIP": [], "UKULELE": []}

                while dips < 4:
                    dip_buy = get_best_orders(dip_book)[0]
                    qty = min(4 - dips, dip_book.buy_orders[dip_buy])
                    dips += qty

                    del dip_book.buy_orders[dip_buy]

                    sell_cost += dip_buy * qty
                    temp_orders["DIP"].append(Order("DIP", dip_buy, -qty))  # sell

                while breads < 2:
                    bread_buy = get_best_orders(bread_book)[0]
                    qty = min(2 - breads, bread_book.buy_orders[bread_buy])
                    breads += qty

                    del bread_book.buy_orders[bread_buy]

                    sell_cost += bread_buy * qty
                    temp_orders["BAGUETTE"].append(Order("BAGUETTE", bread_buy, -qty))  # sell

                uk_buy = get_best_orders(uk_book)[0]
                qty = 1

                sell_cost += uk_buy * qty
                temp_orders["UKULELE"].append(Order("UKULELE", uk_buy, -qty))  # sell

                logger.print("buy basket for", picnic_ask, "sell items for", sell_cost)
                if sell_cost > picnic_ask + offset:
                    result["PICNIC_BASKET"].append(Order("PICNIC_BASKET", picnic_ask, 1))  # buy basket
                    result["DIP"].extend(temp_orders["DIP"])  # sell the rest
                    result["UKULELE"].extend(temp_orders["UKULELE"])
                    result["BAGUETTE"].extend(temp_orders["BAGUETTE"])
                else:
                    break
            except:
                break

        berries_book = state.order_depths["BERRIES"]

        if state.timestamp < 500000:
            if state.position.get('BERRIES', 0) < 250 and min(berries_book.sell_orders.keys()) < 3900:
                result['BERRIES'] = [(
                    Order("BERRIES", min(berries_book.sell_orders.keys()),
                          -berries_book.sell_orders[min(berries_book.sell_orders.keys())]))]
        else:
            if state.position.get('BERRIES', 0) > -250 and max(berries_book.buy_orders.keys()) > 3900:
                result['BERRIES'] = [
                    Order("BERRIES", max(berries_book.buy_orders.keys()),
                          -berries_book.buy_orders[max(berries_book.buy_orders.keys())])]

        logger.flush(state, result)
        return result
