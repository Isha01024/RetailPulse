# src/churn_predictor.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

class ChurnPredictorEngine:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        self.features = ['total_spend', 'tx_count', 'avg_ticket_value', 'recency_days']

    def engineer_churn_features(self, df_sales):
        """Engineers behavioral features based on a rolling customer lifecycle window."""
        max_date = df_sales['date'].max()
        cutoff_date = max_date - pd.Timedelta(days=90)
        
        base_history = df_sales[df_sales['date'] <= cutoff_date]
        active_window = df_sales[df_sales['date'] > cutoff_date]
        
        features_df = base_history.groupby('customer_id').agg(
            total_spend=('total_amount', 'sum'),
            tx_count=('transaction_id', 'count'),
            max_tx_date=('date', 'max')
        ).reset_index()
        
        features_df['avg_ticket_value'] = features_df['total_spend'] / features_df['tx_count']
        features_df['recency_days'] = (cutoff_date - features_df['max_tx_date']).dt.days
        
        active_customers = active_window['customer_id'].unique()
        # Churn defined as 0 transactions within the designated active window
        features_df['is_churned'] = np.where(features_df['customer_id'].isin(active_customers), 0, 1)
        
        return features_df

    def train_pipeline(self, df_sales):
        """Splits data and trains the predictive model."""
        data = self.engineer_churn_features(df_sales)
        if len(data) < 20:
            return "Insufficient profiles to reliably train churn matrix."
            
        X = data[self.features]
        y = data['is_churned']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        accuracy = round(self.model.score(X_test, y_test) * 100, 2)
        return f"Model convergence accomplished. Evaluation Accuracy: {accuracy}%"

    def extract_risk_profiles(self, df_sales):
        """Outputs specific risk probability assignments per active consumer profile."""
        data = self.engineer_churn_features(df_sales)
        X = data[self.features]
        
        data['Churn Risk Probability'] = self.model.predict_proba(X)[:, 1]
        return data[['customer_id', 'total_spend', 'recency_days', 'Churn Risk Probability']].sort_values(by='Churn Risk Probability', ascending=False)
