class Order:
    def __init__(self, id_, price_, quantity_, side_, time_):
        self.id = id_
        self.side = side_
        self.price = price_
        self.quantity = quantity_
        self.time = time_

class OrderBook:
    def __init__(self):
        self.bids = []
        self.asks = []
    
    def add_order(self, order):
        if order.side == 'bid':
            self.bids.append((order.price, order.quantity, order.time))
            self.bids.sort(key=lambda x: -x[0])
        else:
            self.asks.append((order.price, order.quantity, order.time))
            self.aks.sort(key=lambda x: x[0])
    def remove_order(self, order):
        if order.side == 'bid':
            self.bids.remove((order.price, order.quantity, order.time))
        else:
            self.asks.remove((order.price, order.quantity, order.time))

    def get_best_bid_ask(self):
        best_bid = self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return best_bid, best_ask
    def get_position(self, order):
        return 
class Benchmark_mm:
    def __init__(self):
        self.order_book = OrderBook()
        self.bid = None
        self.ask = None
        self.ask_position = 0
        self.bid_position = 0
    def place_order(self, S_t, delta):
        if self.bid:
            self.order_book.remove_order(self.bid)
        if self.ask:
            self.order_book.remove_order(self.ask)
        new_bid = S_t - delta
        new_ask = S_t + delta
        self.bid = new_bid
        self.ask = new_ask
        self.ask_position = self.order_book.get_position(self.ask)
        self.bid_position = self.order_book.get_position(self.bid)
