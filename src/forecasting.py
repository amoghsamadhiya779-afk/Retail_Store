import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RetailForecaster:
    """
    Handles Time-Series Forecasting using Facebook Prophet.
    Incorporates Holidays and External Regressors (CPI, Fuel, etc.).
    """
    def __init__(self, df):
        self.df = df.copy()
        self.model = None
        self.forecast = None

    def prepare_data(self, store_id, dept_id):
        """
        Prepares data for a specific Store-Dept combination.
        Prophet requires columns named 'ds' (Date) and 'y' (Target).
        """
        print(f"📉 Preparing data for Store {store_id}, Dept {dept_id}...")
        
        # Filter
        subset = self.df[(self.df['Store'] == store_id) & (self.df['Dept'] == dept_id)].copy()
        
        if subset.empty:
            raise ValueError(f"No data found for Store {store_id} Dept {dept_id}")

        # Rename for Prophet
        subset = subset.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
        
        # Sort by date
        subset = subset.sort_values('ds')
        
        return subset

    def train_model(self, train_df, regressors=None):
        """
        Trains the Prophet model.
        """
        if train_df.empty:
            raise ValueError("Training data is empty. Cannot train model.")

        print("🧠 Training Prophet Model...")
        
        # Initialize Prophet
        # We enable yearly seasonality (Retail) and weekly (Shopping habits)
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95  # 95% Confidence Interval
        )
        
        # Add Holidays (Built-in country holidays)
        self.model.add_country_holidays(country_name='US')
        
        # Add External Regressors (Economic Factors)
        if regressors:
            print(f"   Adding regressors: {regressors}")
            for reg in regressors:
                self.model.add_regressor(reg)
                
        # Fit Model
        self.model.fit(train_df)
        return self.model

    def predict(self, periods=12, future_regressors_df=None):
        """
        Generates forecast for N periods into the future.
        If using regressors, future_regressors_df must provide values for the future dates.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print(f"🔮 Forecasting {periods} weeks ahead...")
        
        # Create future dataframe placeholder
        # Prophet automatically detects frequency from history, but specifying freq='W' ensures weekly
        future = self.model.make_future_dataframe(periods=periods, freq='W')
        
        # If we have regressors, we need to merge their future values
        if future_regressors_df is not None:
            # Ensure 'ds' column exists
            if 'ds' not in future_regressors_df.columns:
                 if 'Date' in future_regressors_df.columns:
                     future_regressors_df = future_regressors_df.rename(columns={'Date': 'ds'})
            
            # Merge
            future = future.merge(future_regressors_df, on='ds', how='left')
            
            # Forward fill missing economic data for the future (assumption: economy stays stable)
            future = future.ffill()
            
            # Check for NaNs (Prophet crashes on NaNs in regressors)
            if future.isnull().any().any():
                future = future.fillna(method='bfill').fillna(0)

        self.forecast = self.model.predict(future)
        return self.forecast

    def evaluate(self, test_df):
        """
        Calculates RMSE and MAE against the Test set.
        """
        if self.forecast is None:
            raise ValueError("No forecast found. Call predict() first.")

        # Inner join to align dates
        comparison = test_df.merge(self.forecast[['ds', 'yhat']], on='ds', how='inner')
        
        if comparison.empty:
            print("⚠️ Warning: No overlapping dates found between Forecast and Test Data.")
            return np.nan, np.nan, comparison

        rmse = np.sqrt(mean_squared_error(comparison['y'], comparison['yhat']))
        mae = mean_absolute_error(comparison['y'], comparison['yhat'])
        
        print(f"📊 Evaluation Results:")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAE:  {mae:.2f}")
        
        return rmse, mae, comparison

    def plot_components(self):
        """
        Visualizes the trend, seasonality, and holidays.
        """
        if self.model is None or self.forecast is None:
            print("⚠️ Model or Forecast missing. Cannot plot components.")
            return
            
        fig = self.model.plot_components(self.forecast)
        plt.show()