class OrderBook
    def __init__(self):
        self.bid = []
        self.ask = []
    
    def add_bid(self, price, quantity):
        self.bid.append((price, quantity))
    
    def add_ask(self, price, quantity):
        self.ask.append((price, quantity))
    
