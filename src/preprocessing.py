import os
import pandas as pd
import numpy as np

class RetailPreprocessor:
    """
    Handles cleaning, merging, and feature engineering for Retail Data.
    """
    def __init__(self, df=None):
        self.df = df

    def load_raw_data(self, data_path="data/raw"):
        """
        Loads raw datasets from the specified path.
        Defines default paths for Sales, Stores, and Features.
        """
        print(f"📂 Loading raw data from: {data_path}")
        
        # Define paths for raw data
        sales_path = os.path.join(data_path, "Sales.csv")
        # Check for alternative naming (train.csv) commonly used in Kaggle
        if not os.path.exists(sales_path):
             if os.path.exists(os.path.join(data_path, "Sales.csv")):
                 sales_path = os.path.join(data_path, "Sales.csv")
             else:
                 sales_path = os.path.join(data_path, " Sales.csv")

        stores_path = os.path.join(data_path, "Stores.csv")
        if not os.path.exists(stores_path):
            stores_path = os.path.join(data_path, "stores.csv")

        features_path = os.path.join(data_path, "Features.csv")
        if not os.path.exists(features_path):
            features_path = os.path.join(data_path, "features.csv")

        # Load CSVs
        try:
            sales = pd.read_csv(sales_path)
            stores = pd.read_csv(stores_path)
            features = pd.read_csv(features_path)
            print("   ✅ Raw datasets loaded successfully.")
            
            # Automatically merge after loading
            return self.merge_datasets(sales, stores, features)
            
        except FileNotFoundError as e:
            print(f"❌ Error loading raw data: {e}")
            return None

    def merge_datasets(self, sales, stores, features):
        """
        Merges the raw datasets: Sales + Stores + Features.
        """
        print("🔄 Merging datasets inside Preprocessor...")
        
        # Working on copies to avoid side-effects on original dataframes
        sales = sales.copy()
        features = features.copy()
        stores = stores.copy()

        # 1. Handle Dates with robustness
        if 'Date' in sales.columns and not pd.api.types.is_datetime64_any_dtype(sales['Date']):
            sales['Date'] = pd.to_datetime(sales['Date'], dayfirst=True, errors='coerce')
        
        if 'Date' in features.columns and not pd.api.types.is_datetime64_any_dtype(features['Date']):
            features['Date'] = pd.to_datetime(features['Date'], dayfirst=True, errors='coerce')

        # 2. Handle IsHoliday (String/Int -> Boolean)
        bool_map = {'True': True, 'False': False, 'true': True, 'false': False, 1: True, 0: False, True: True, False: False}
        
        if 'IsHoliday' in sales.columns:
            sales['IsHoliday'] = sales['IsHoliday'].map(bool_map).astype(bool)
        if 'IsHoliday' in features.columns:
            features['IsHoliday'] = features['IsHoliday'].map(bool_map).astype(bool)
        
        # 3. Merge Datasets
        # Merge Sales + Stores (Left join on Store)
        df_merged = sales.merge(stores, on='Store', how='left')
        
        # Merge Result + Features (Left join on Store, Date, IsHoliday)
        df_merged = df_merged.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
        
        self.df = df_merged
        print(f"   ✅ Merge Complete. Final Shape: {self.df.shape}")
        return self.df

    def clean_data(self):
        """
        Executes standard cleaning steps.
        """
        if self.df is None:
            raise ValueError("❌ DataFrame is empty! Call merge_datasets() first.")

        print("🧹 Starting Preprocessing...")

        # 1. Date Conversion (Safety check)
        if 'Date' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'], dayfirst=True, errors='coerce')
        
        # Drop rows where Date failed to parse
        if self.df['Date'].isnull().sum() > 0:
            print(f"   ⚠️ Warning: Dropping {self.df['Date'].isnull().sum()} rows with invalid dates.")
            self.df = self.df.dropna(subset=['Date'])

        # 2. Sort by Date (Crucial for Time Series features)
        self.df = self.df.sort_values(by=['Date', 'Store', 'Dept'])

        # 3. Handle MarkDowns 
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        existing_markdowns = [col for col in markdown_cols if col in self.df.columns]
        if existing_markdowns:
            self.df[existing_markdowns] = self.df[existing_markdowns].fillna(0)

        # 4. Handle Economic Indicators 
        if 'CPI' in self.df.columns:
            self.df['CPI'] = self.df['CPI'].ffill() 
        if 'Unemployment' in self.df.columns:
            self.df['Unemployment'] = self.df['Unemployment'].ffill()

        # 5. Convert Boolean IsHoliday to Int (1/0) for ML models
        if 'IsHoliday' in self.df.columns:
            self.df['IsHoliday'] = self.df['IsHoliday'].astype(int)
        
        # 6. Handle Negative Sales (Returns)
        if 'Weekly_Sales' in self.df.columns:
            neg_sales = self.df[self.df['Weekly_Sales'] < 0].shape[0]
            if neg_sales > 0:
                print(f"   ⚠️ Warning: Dataset contains {neg_sales} records with negative sales (Returns).")

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

if __name__ == "__main__":
    # Test execution
    processor = RetailPreprocessor()
    # Define paths of raw data implicitly by calling load_raw_data with default
    processor.load_raw_data(data_path="data/raw")
    if processor.df is not None:
        processor.clean_data()
        processor.feature_engineering()
        processor.save_data()