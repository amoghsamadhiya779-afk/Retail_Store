import pandas as pd
import numpy as np

class RetailPreprocessor:
    """
    Handles cleaning and feature engineering.
    """
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        """
        Executes standard cleaning steps:
        1. Convert Date to datetime.
        2. Fill missing MarkDown values with 0.
        3. Fill missing CPI/Unemployment with forward fill.
        4. Encode categorical IsHoliday.
        """
        print("🧹 Starting Preprocessing...")

        # 1. Date Conversion
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # 2. Sort by Date (Crucial for Time Series)
        self.df = self.df.sort_values(by=['Date', 'Store', 'Dept'])

        # 3. Handle MarkDowns (Null implies no markdown active -> 0)
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        self.df[markdown_cols] = self.df[markdown_cols].fillna(0)

        # 4. Handle Economic Indicators (Forward fill for missing weeks)
        self.df['CPI'] = self.df['CPI'].fillna(method='ffill')
        self.df['Unemployment'] = self.df['Unemployment'].fillna(method='ffill')

        # 5. Convert Boolean to Int
        if 'IsHoliday' in self.df.columns:
            self.df['IsHoliday'] = self.df['IsHoliday'].astype(int)

        print("✅ Preprocessing Complete.")
        return self.df

    def feature_engineering(self):
        """
        Adds basic time-series features.
        """
        # Extract Date components
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Week'] = self.df['Date'].dt.isocalendar().week
        
        return self.df