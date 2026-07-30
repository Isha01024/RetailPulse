# src/forecaster.py
import pandas as pd
import numpy as np
from prophet import Prophet

class DemandForecaster:
    @staticmethod
    def forecast_category_demand(df_sales, category, forecast_days=30):
        """Extracts timeline data per category and generates a 30-day ahead forecast."""
        cat_data = df_sales[df_sales['product_category'] == category].copy()
        daily_series = cat_data.groupby('date')['quantity'].sum().reset_index()
        daily_series.columns = ['ds', 'y']
        
        if len(daily_series) < 10:
            future_dates = pd.date_range(start=df_sales['date'].max() + pd.Timedelta(days=1), periods=forecast_days)
            return pd.DataFrame({'ds': future_dates, 'yhat': [10] * forecast_days})
            
        # Initialize Prophet with explicit retail-centric seasonality variables
        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
        model.fit(daily_series)
        
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat']].tail(forecast_days)

    @staticmethod
    def optimize_inventory(forecast_qty, current_stock, reorder_pt, lead_time):
        """Operations logic evaluating run-rates and procurement triggers."""
        projected_demand = max(0, int(forecast_qty.sum()))
        suggested_order = max(0, (projected_demand + (reorder_pt * 1.2)) - current_stock)
        
        status = "Healthy Stock Range"
        if current_stock <= reorder_pt:
            status = "CRITICAL REORDER TRIGGERED"
        elif current_stock > (projected_demand * 1.5):
            status = "OVERSTOCKED RISK"
            
        return {
            "Projected Demand (30D)": projected_demand,
            "Suggested Procurement": int(suggested_order),
            "Stock Evaluation Status": status
        }
