# src/data_processor.py
import pandas as pd
import numpy as np

class DataWarehousePipeline:
    def __init__(self, sales_path='data/raw_sales.csv', cust_path='data/raw_customers.csv', inv_path='data/raw_inventory.csv'):
        self.sales_path = sales_path
        self.cust_path = cust_path
        self.inv_path = inv_path

    def load_and_clean(self):
        """Loads and enforces type casting schemas across standard business frames."""
        sales = pd.read_csv(self.sales_path)
        customers = pd.read_csv(self.cust_path)
        inventory = pd.read_csv(self.inv_path)
        
        # Enforce analytical date hierarchies
        sales['date'] = pd.to_datetime(sales['date'])
        customers['signup_date'] = pd.to_datetime(customers['signup_date'])
        
        # Prevent row fragmentation from missing structural primary keys
        sales.dropna(subset=['customer_id', 'total_amount'], inplace=True)
        
        return sales, customers, inventory

    def get_master_unified_frame(self):
        """Combines the raw files into an integrated master processing dataframe."""
        sales, customers, inventory = self.load_and_clean()
        merged = pd.merge(sales, customers, on='customer_id', how='left')
        merged = pd.merge(merged, inventory, on='product_category', how='left')
        return merged
