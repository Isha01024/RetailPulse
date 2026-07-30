import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data():
    os.makedirs('data', exist_ok=True)
    np.random.seed(42)
    
    # 1. Customers Baseline Core (1,000 distinct profiles)
    customer_ids = [f"CUST-{i:04d}" for i in range(1, 1001)]
    genders = np.random.choice(['M', 'F'], size=1000, p=[0.45, 0.55])
    ages = np.random.randint(18, 70, size=1000)
    income = np.random.choice(['Low', 'Medium', 'High'], size=1000, p=[0.3, 0.5, 0.2])
    
    df_customers = pd.DataFrame({
        'customer_id': customer_ids,
        'gender': genders,
        'age': ages,
        'income_bracket': income,
        'signup_date': [datetime(2024, 1, 1) + timedelta(days=int(np.random.randint(0, 365))) for _ in range(1000)]
    })
    df_customers.to_csv('data/raw_customers.csv', index=False)
    
    # 2. Sales Log (Simulating transactional demand data over ~2.3 Years)
    products = ['Electronics', 'Apparel', 'Home & Kitchen', 'Groceries', 'Beauty']
    product_weights = [0.25, 0.30, 0.15, 0.20, 0.10]
    
    start_date = datetime(2024, 1, 1)
    date_list = [start_date + timedelta(days=x) for x in range(850)] 
    
    sales_records = []
    for dt in date_list:
        # Seasonality scaling factors for year-end spikes
        num_transactions = int(np.random.randint(100, 200) * (1.3 if dt.month in [11, 12] else 1.0))
        for _ in range(num_transactions):
            prod = np.random.choice(products, p=product_weights)
            quant = int(np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.25, 0.15, 0.07, 0.03]))
            
            base_prices = {'Electronics': 500, 'Apparel': 45, 'Home & Kitchen': 120, 'Groceries': 12, 'Beauty': 35}
            price = base_prices[prod] * (1 + np.random.uniform(-0.1, 0.1))
            
            sales_records.append({
                'transaction_id': f"TX-{len(sales_records)+100000:06d}",
                'date': dt.strftime('%Y-%m-%d'),
                'customer_id': np.random.choice(customer_ids),
                'product_category': prod,
                'quantity': quant,
                'unit_price': round(price, 2),
                'total_amount': round(quant * price, 2)
            })
            
    df_sales = pd.DataFrame(sales_records)
    df_sales.to_csv('data/raw_sales.csv', index=False)
    
    # 3. Inventory Management System Baseline
    inventory_records = []
    for prod in products:
        base_stock = {'Electronics': 200, 'Apparel': 1200, 'Home & Kitchen': 450, 'Groceries': 3000, 'Beauty': 600}
        inventory_records.append({
            'product_category': prod,
            'current_stock_level': base_stock[prod],
            'reorder_point': int(base_stock[prod] * 0.25),
            'safety_stock': int(base_stock[prod] * 0.10),
            'lead_time_days': np.random.choice([3, 5, 7, 10])
        })
    df_inventory = pd.DataFrame(inventory_records)
    df_inventory.to_csv('data/raw_inventory.csv', index=False)
    print("✓ Data generation completed. Raw CSV assets saved to /data folder.")

if __name__ == "__main__":
    generate_mock_data()
