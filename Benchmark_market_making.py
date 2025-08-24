import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class Order:
    def __init__(self, price, quantity, side, time):
        self.price = price
        self.quantity = quantity
        self.side = side
        self.time = time

class MarketOrder:
    def __init__(self, side, quantity, price=None):
        self.side = side
        self.quantity = quantity
        self.price = price

class OrderBook:
    def __init__(self):
        self.bids = []
        self.asks = [] 
    
    def add_order(self, order):
        if order.side == 'bid':
            self.bids.append((order.price, order.quantity, order.time, order))
            self.bids.sort(key=lambda x: -x[0])
        else:
            self.asks.append((order.price, order.quantity, order.time, order))
            self.asks.sort(key=lambda x: x[0])
    
    def remove_order(self, order):
        if order.side == 'bid':
            for i, (p, q, t, o) in enumerate(self.bids):
                if o == order:
                    self.bids.pop(i)
                    return
        else:
            for i, (p, q, t, o) in enumerate(self.asks):
                if o == order:
                    self.asks.pop(i)
                    return
    
    def get_best_bid_ask(self):
        best_bid = self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return best_bid, best_ask
    
    def get_position(self, order):
        if order.side == 'bid':
            for i, (p, q, t, o) in enumerate(self.bids):
                if o == order:
                    return i
        else:
            for i, (p, q, t, o) in enumerate(self.asks):
                if o == order:
                    return i
        return None
class Benchmark_mm:
    def __init__(self, order_book_, inventory_limit=10):
        self.order_book = order_book_
        self.bid = None
        self.ask = None
        self.ask_position = 0
        self.bid_position = 0
        self.inventory = 0
        self.inventory_limit = inventory_limit
        self.pnl = 0
    
    def place_order(self, S_t, delta, quantity, time):
        if self.inventory >= self.inventory_limit:
            self.bid = None
        else:
            self.bid = Order(S_t - delta, quantity, "bid", time)
            self.order_book.add_order(self.bid)
        
        if self.inventory <= -self.inventory_limit:
            self.ask = None
        else:
            self.ask = Order(S_t + delta, quantity, "ask", time)
            self.order_book.add_order(self.ask)
        
        if self.bid:
            self.bid_position = self.order_book.get_position(self.bid)
        if self.ask:
            self.ask_position = self.order_book.get_position(self.ask)

def simulate_order_execution(order_book, market_orders, mm):
    """Simule l'exécution avec mise à jour en temps réel de l'inventaire"""
    executed_orders = []
    
    for mo in market_orders:
        if mo.side == 'buy' and order_book.asks:
            qty_remaining = mo.quantity
            while qty_remaining > 0 and order_book.asks:
                if mm.inventory <= -mm.inventory_limit:
                    break
                
                best_ask_price, best_ask_qty, best_ask_time, best_ask_order = order_book.asks[0]
                
                if mo.price is None or best_ask_price <= mo.price:
                    max_sell = mm.inventory_limit + mm.inventory
                    fill_qty = min(best_ask_qty, qty_remaining, max_sell)
                    
                    if fill_qty > 0:
                        executed_orders.append(('ask', best_ask_price, fill_qty, best_ask_order))
                        
                        mm.inventory -= fill_qty
                        mm.pnl += best_ask_price * fill_qty
                        
                        if best_ask_qty == fill_qty:
                            order_book.asks.pop(0)
                        else:
                            order_book.asks[0] = (best_ask_price, best_ask_qty - fill_qty, best_ask_time, best_ask_order)
                        
                        qty_remaining -= fill_qty
                        
                        print(f"Vente exécutée: {fill_qty} @ {best_ask_price:.3f} | Inventory: {mm.inventory}")
                    else:
                        break
                else:
                    break
        
        elif mo.side == 'sell' and order_book.bids:
            qty_remaining = mo.quantity
            while qty_remaining > 0 and order_book.bids:
                if mm.inventory >= mm.inventory_limit:
                    break
                
                best_bid_price, best_bid_qty, best_bid_time, best_bid_order = order_book.bids[0]
                
                if mo.price is None or best_bid_price >= mo.price:
                    max_buy = mm.inventory_limit - mm.inventory
                    fill_qty = min(best_bid_qty, qty_remaining, max_buy)
                    
                    if fill_qty > 0:
                        executed_orders.append(('bid', best_bid_price, fill_qty, best_bid_order))
                        
                        mm.inventory += fill_qty
                        mm.pnl -= best_bid_price * fill_qty
                        
                        if best_bid_qty == fill_qty:
                            order_book.bids.pop(0)
                        else:
                            order_book.bids[0] = (best_bid_price, best_bid_qty - fill_qty, best_bid_time, best_bid_order)
                        
                        qty_remaining -= fill_qty
                        
                        print(f"Achat exécuté: {fill_qty} @ {best_bid_price:.3f} | Inventory: {mm.inventory}")
                    else:
                        break
                else:
                    break
    
    return executed_orders

def generate_market_orders(S_t, volatility, intensity=3):
    orders = []
    n_orders = np.random.poisson(intensity)
    
    for _ in range(n_orders):
        side = 'buy' if np.random.random() > 0.5 else 'sell'
        quantity = np.random.randint(1, 5)
        price_ref = S_t * (1 + np.random.normal(0, volatility))
        orders.append(MarketOrder(side, quantity, price_ref))
    
    return orders



if __name__ == '__main__':
    N = 1000
    T = 1
    sigma = 0.02
    dt = T / N
    delta = 0.02
    
    dW = np.random.normal(0, np.sqrt(dt), N)
    S = np.cumsum(dW) * sigma
    
    order_book = OrderBook()
    mm = Benchmark_mm(order_book)
    
    pnl_history = []
    
    for t in range(N):
        S_t = S[t]

        # 1. Le market-maker place ses ordres
        mm.place_order(S_t, delta, 1, t)
        
        # 2. Génération des ordres de marché
        market_orders = generate_market_orders(S_t, sigma)
        
        # 3. Exécution des ordres
        executed = simulate_order_execution(order_book, market_orders, mm)
        
        
        pnl_history.append(mm.pnl)
        inventory_risk_cost = 0.0001 * mm.inventory**2 * dt
        mm.pnl -= inventory_risk_cost
        print(f"Time {t}: PnL = {mm.pnl:.2f}, Inventory = {mm.inventory}")

    # Résultats finaux
    print(f"\n=== RÉSULTATS FINAUX ===")
    print(f"Final PnL: {mm.pnl:.2f}")
    print(f"Final Inventory: {mm.inventory}")
    print(f"PnL Std: {np.std(pnl_history):.2f}")

        
