import os
import pandas as pd
import numpy as np

class RetailPreprocessor:
    """
    Handles cleaning, merging, and feature engineering for Retail Data.
    """
    def __init__(self, df=None):
        self.df = df

    def merge_datasets(self, sales, stores, features):
        """
        Merges the raw datasets: Sales + Stores + Features.
        """
        print("🔄 Merging datasets inside Preprocessor...")
        
        # 1. Ensure Date formats match before merging (Crucial fix)
        # Using to_datetime ensures that string '2010-02-05' matches datetime objects
        if 'Date' in sales.columns:
            sales['Date'] = pd.to_datetime(sales['Date'])
        if 'Date' in features.columns:
            features['Date'] = pd.to_datetime(features['Date'])

        # 2. Ensure IsHoliday is boolean in both to prevent mismatch ('True' vs True)
        if 'IsHoliday' in sales.columns:
            sales['IsHoliday'] = sales['IsHoliday'].astype(bool)
        if 'IsHoliday' in features.columns:
            features['IsHoliday'] = features['IsHoliday'].astype(bool)
        
        # Merge Sales + Stores (Left join on Store)
        df_merged = sales.merge(stores, on='Store', how='left')
        
        # Merge Result + Features (Left join on Store, Date, IsHoliday)
        df_merged = df_merged.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
        
        self.df = df_merged
        print(f"   ✅ Merge Complete. Final Shape: {self.df.shape}")
        return self.df

    def clean_data(self):
        """
        Executes standard cleaning steps:
        1. Convert Date to datetime.
        2. Sort by Date/Store/Dept.
        3. Fill missing MarkDown values with 0 (Standard Retail Practice).
        4. Fill missing CPI/Unemployment with forward fill (Time Series).
        5. Encode categorical IsHoliday.
        """
        if self.df is None:
            raise ValueError("❌ DataFrame is empty! Call merge_datasets() first.")

        print("🧹 Starting Preprocessing...")

        # 1. Date Conversion
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # 2. Sort by Date (Crucial for Time Series features later)
        self.df = self.df.sort_values(by=['Date', 'Store', 'Dept'])

        # 3. Handle MarkDowns 
        # Logic: If MarkDown is NaN, it means no promotion was active, so we fill with 0.
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        existing_markdowns = [col for col in markdown_cols if col in self.df.columns]
        
        if existing_markdowns:
            self.df[existing_markdowns] = self.df[existing_markdowns].fillna(0)

        # 4. Handle Economic Indicators 
        # Logic: Economic indicators don't change daily, often weekly/monthly. 
        # If missing, we assume the previous known value (Forward Fill).
        if 'CPI' in self.df.columns:
            self.df['CPI'] = self.df['CPI'].ffill() 
        if 'Unemployment' in self.df.columns:
            self.df['Unemployment'] = self.df['Unemployment'].ffill()

        # 5. Convert Boolean IsHoliday to Int (1/0)
        if 'IsHoliday' in self.df.columns:
            self.df['IsHoliday'] = self.df['IsHoliday'].astype(int)
        
        # 6. Handle Negative Sales (Returns)
        if 'Weekly_Sales' in self.df.columns:
            neg_sales = self.df[self.df['Weekly_Sales'] < 0].shape[0]
            if neg_sales > 0:
                print(f"⚠️ Warning: Dataset contains {neg_sales} records with negative sales (Returns).")

        print("✅ Cleaning Complete.")
        return self.df

    def feature_engineering(self):
        """
        Adds basic time-series features useful for forecasting.
        """
        if self.df is None:
             raise ValueError("Dataframe not initialized.")

        print("⚙️ Engineering Features...")
        
        # Extract Date components
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Week'] = self.df['Date'].dt.isocalendar().week.astype(int)
        self.df['Quarter'] = self.df['Date'].dt.quarter
        
        # Day of Year (useful for seasonality)
        self.df['DayOfYear'] = self.df['Date'].dt.dayofyear
        
        return self.df

    def save_data(self, output_path="data/processed/clean_retail_data.csv"):
        """
        Saves the processed data to the specified path.
        """
        if self.df is None:
             raise ValueError("Dataframe is empty. Cannot save.")
             
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        self.df.to_csv(output_path, index=False)
        print(f"✅ Processed data saved successfully to: {output_path}")