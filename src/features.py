import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Handles advanced feature engineering for Retail Forecasting.
    Includes: Lags, Rolling Windows, Holiday Flags, and Encoding.
    """
    def __init__(self, df):
        self.df = df

    def create_time_series_features(self):
        """
        Creates Lag and Rolling features. 
        CRITICAL: Must be grouped by Store and Dept to prevent data leakage between departments.
        """
        print("⚙️ Generatng Time-Series features (Lags & Rolling)... this may take a moment.")
        
        # Ensure data is sorted
        self.df = self.df.sort_values(by=['Store', 'Dept', 'Date'])
        
        # 1. Lag Features (Past Sales)
        # Lag 1: Sales 1 week ago (Auto-regressive)
        # Lag 52: Sales 1 year ago (Seasonality)
        self.df['Lag_1'] = self.df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
        self.df['Lag_52'] = self.df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(52)
        
        # 2. Rolling Window Features (Trend)
        # 4-week moving average
        self.df['Rolling_Mean_4'] = self.df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(window=4).mean())
        # 4-week standard deviation (Volatility)
        self.df['Rolling_Std_4'] = self.df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(window=4).std())

        # Fill NaN values created by lags (usually with 0 or drop)
        # We will fill with 0 to keep the data size, or use backfill
        self.df = self.df.fillna(0)
        
        return self.df

    def create_special_holiday_features(self):
        """
        The generic 'IsHoliday' is not enough. 
        We need to identify specific holidays like Super Bowl, Christmas, etc.
        """
        print("🎄 Marking specific holidays...")
        
        # Dates based on the specific Walmart dataset timeline (2010-2013)
        super_bowl_dates = pd.to_datetime(['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08'])
        labor_day_dates = pd.to_datetime(['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06'])
        thanksgiving_dates = pd.to_datetime(['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29'])
        christmas_dates = pd.to_datetime(['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'])
        
        self.df['Is_SuperBowl'] = self.df['Date'].isin(super_bowl_dates).astype(int)
        self.df['Is_LaborDay'] = self.df['Date'].isin(labor_day_dates).astype(int)
        self.df['Is_Thanksgiving'] = self.df['Date'].isin(thanksgiving_dates).astype(int)
        self.df['Is_Christmas'] = self.df['Date'].isin(christmas_dates).astype(int)
        
        return self.df

    def encode_categorical(self):
        """
        Encodes Store Type (A, B, C) and other categorical variables.
        """
        print("🔠 Encoding categorical variables...")
        
        # Store Type: One-Hot Encoding or Ordinal?
        # Let's use Ordinal mapping for simplicity based on median size usually: A > B > C
        type_map = {'A': 3, 'B': 2, 'C': 1}
        if 'Type' in self.df.columns:
            self.df['Type_Encoded'] = self.df['Type'].map(type_map).fillna(0)
            
        return self.df

    def create_interaction_features(self):
        """
        Creates interaction features like Markdown Sum.
        """
        print("🔗 Creating interaction features...")
        
        # Total Markdown Impact
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        self.df['Total_MarkDown'] = self.df[markdown_cols].sum(axis=1)
        
        return self.df