# src/segmentor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class CustomerSegmentor:
    def __init__(self):
        self.scaler = StandardScaler()
        # Initializing K-Means clustering with 4 actionable operational profiles
        self.model = KMeans(n_clusters=4, random_state=42, n_init=10)

    def compute_rfm_segments(self, df_sales):
        """Calculates Recency, Frequency, and Monetary scores and runs clustering."""
        max_date = df_sales['date'].max()
        
        rfm = df_sales.groupby('customer_id').agg({
            'date': lambda x: (max_date - x.max()).days,
            'transaction_id': 'count',
            'total_amount': 'sum'
        }).rename(columns={'date': 'Recency', 'transaction_id': 'Frequency', 'total_amount': 'Monetary'})
        
        # Scale inputs for fair distance measurements
        features = ['Recency', 'Frequency', 'Monetary']
        scaled_features = self.scaler.fit_transform(rfm[features])
        
        rfm['Cluster'] = self.model.fit_predict(scaled_features)
        
        # Strategic segmentation business rules
        cluster_mapping = {
            0: "Core Value Leaders", 
            1: "At Risk / Inactive", 
            2: "High-Volume Buyers", 
            3: "Emerging Regulars"
        }
        rfm['Segment Name'] = rfm['Cluster'].map(cluster_mapping)
        return rfm
