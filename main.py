import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

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

class Benchmark_mm:
    def __init__(self, order_book_, inventory_limit=10, A=140, k=1.5):
        self.order_book = order_book_
        self.bid = None
        self.ask = None
        self.inventory = 0
        self.pnl = 0
        self.inventory_limit = inventory_limit
        self.A = A
        self.k = k
    
    def place_order(self, S_t, delta, quantity, time, dt):
        if self.bid:
            self.order_book.remove_order(self.bid)
        if self.ask:
            self.order_book.remove_order(self.ask)
        
        if self.inventory < self.inventory_limit:
            bid_price = S_t - delta
            self.bid = Order(bid_price, quantity, "bid", time)
            self.order_book.add_order(self.bid)
        else:
            self.bid = None
        
        if self.inventory > -self.inventory_limit:
            ask_price = S_t + delta
            self.ask = Order(ask_price, quantity, "ask", time)
            self.order_book.add_order(self.ask)
        else:
            self.ask = None
        
        if self.bid:
            delta_b = S_t - bid_price
            lambda_b = self.A * math.exp(-self.k * delta_b)
            p_bid = 1 - math.exp(-lambda_b * dt)
            if np.random.random() < p_bid:
                self.inventory -= quantity
                self.pnl += bid_price * quantity
        
        if self.ask:
            delta_a = ask_price - S_t
            lambda_a = self.A * math.exp(-self.k * delta_a)
            p_ask = 1 - math.exp(-lambda_a * dt)
            if np.random.random() < p_ask:
                self.inventory += quantity
                self.pnl -= ask_price * quantity

class Avellaneda_Stoikov:
    def __init__(self, order_book_, gamma=0.1, k=1.5, A=140, sigma=0.02, T=1, N=1000, inventory_limit=10):
        self.order_book = order_book_
        self.bid = None
        self.ask = None
        self.inventory = 0
        self.pnl = 0
        self.gamma = gamma
        self.k = k
        self.A = A
        self.sigma = sigma
        self.T = T
        self.N = N
        self.inventory_limit = inventory_limit
    
    def reservation_price(self, S_t, t):
        time_to_maturity = self.T - t/self.N
        r_t = S_t - self.inventory * self.gamma * self.sigma ** 2 * time_to_maturity
        spread = self.gamma * self.sigma ** 2 * time_to_maturity + (2/self.gamma) * math.log(1 + self.gamma/self.k)
        
        ask_price = r_t + spread/2
        bid_price = r_t - spread/2
        
        return bid_price, ask_price, spread
    
    def place_order(self, S_t, t, quantity, dt):
        if self.bid:
            self.order_book.remove_order(self.bid)
        if self.ask:
            self.order_book.remove_order(self.ask)
        
        bid_price, ask_price, spread = self.reservation_price(S_t, t)

        if self.inventory < self.inventory_limit:
            self.bid = Order(bid_price, quantity, "bid", t)
            self.order_book.add_order(self.bid)
        else:
            self.bid = None
        
        if self.inventory > -self.inventory_limit:
            self.ask = Order(ask_price, quantity, "ask", t)
            self.order_book.add_order(self.ask)
        else:
            self.ask = None
        
        dNa, dNb = 0, 0
        
        if self.ask:
            delta_a = ask_price - S_t
            lambda_a = self.A * math.exp(-self.k * delta_a)
            p_ask = 1 - math.exp(-lambda_a * dt)
            if np.random.random() < p_ask:
                dNa = quantity
        
        if self.bid:
            delta_b = S_t - bid_price
            lambda_b = self.A * math.exp(-self.k * delta_b)
            p_bid = 1 - math.exp(-lambda_b * dt)
            if np.random.random() < p_bid:
                dNb = quantity
        
        self.inventory += dNb - dNa
        self.pnl += ask_price * dNa - bid_price * dNb

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
    # Paramètres
    N = 1000
    T = 1
    sigma = 2
    dt = T / N
    iterations = 1000
    
    pnl_history_as = []
    pnl_history_benchmark = []
    inventory_history_as = []
    inventory_history_benchmark = []
    
    for iter in range(iterations):
        print(f"Simulation {iter+1}/{iterations}")
        
        # Génération du prix
        dW = np.random.normal(0, np.sqrt(dt), N)
        S = 100 + np.cumsum(dW) * sigma * 100
        
        # Benchmark
        order_book_bench = OrderBook()
        mm_bench = Benchmark_mm(order_book_bench, A=140, k=1.5)
        pnl_bench = []
        inv_bench = []
        
        # Avellaneda-Stoikov
        order_book_as = OrderBook()
        mm_as = Avellaneda_Stoikov(order_book_as, gamma=0.1, k=1.5, A=140, 
                                 sigma=sigma, T=T, N=N)
        pnl_as = []
        inv_as = []
        
        for t in range(N):
            S_t = S[t]
            
            # Benchmark
            mm_bench.place_order(S_t, 0.5, 1, t, dt)
            inventory_risk_cost = 0.0001 * mm_bench.inventory**2 * dt
            mm_bench.pnl -= inventory_risk_cost
            pnl_bench.append(mm_bench.pnl)
            inv_bench.append(mm_bench.inventory)
            
            # Avellaneda-Stoikov
            mm_as.place_order(S_t, t, 1, dt)
            inventory_risk_cost = 0.0001 * mm_as.inventory**2 * dt
            mm_as.pnl -= inventory_risk_cost
            pnl_as.append(mm_as.pnl)
            inv_as.append(mm_as.inventory)
        
        pnl_history_benchmark.append(pnl_bench)
        pnl_history_as.append(pnl_as)
        inventory_history_benchmark.append(inv_bench)
        inventory_history_as.append(inv_as)
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # PnL
    ax1.plot(np.mean(pnl_history_benchmark, axis=0), label='Benchmark (spread fixe)', linewidth=2)
    ax1.plot(np.mean(pnl_history_as, axis=0), label='Avellaneda-Stoikov', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('PnL')
    ax1.legend()
    ax1.set_title('Comparison: PnL Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Inventory
    ax2.plot(np.mean(inventory_history_benchmark, axis=0), label='Benchmark', linewidth=2)
    ax2.plot(np.mean(inventory_history_as, axis=0), label='Avellaneda-Stoikov', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Inventory')
    ax2.legend()
    ax2.set_title('Inventory Management')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    final_pnl_bench = [x[-1] for x in pnl_history_benchmark]
    final_pnl_as = [x[-1] for x in pnl_history_as]
    
    print(f"\n=== RÉSULTATS FINAUX ===")
    print(f"Benchmark - Mean PnL: {np.mean(final_pnl_bench):.2f}, Std: {np.std(final_pnl_bench):.2f}")
    print(f"Avellaneda-Stoikov - Mean PnL: {np.mean(final_pnl_as):.2f}, Std: {np.std(final_pnl_as):.2f}")
    print(f"Improvement: {((np.mean(final_pnl_as) - np.mean(final_pnl_bench)) / np.mean(final_pnl_bench) * 100):.1f}%")