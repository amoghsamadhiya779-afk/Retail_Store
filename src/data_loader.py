import pandas as pd
import os

class RetailDataLoader:
    """
    Handles loading and merging of the Retail Dataset.
    """
    def __init__(self, data_path="data/raw"):
        self.data_path = data_path

    def load_data(self):
        """
        Loads Stores, Features, and Train datasets and merges them.
        """
        print("⏳ Loading datasets...")
        
        # Define paths
        stores_path = os.path.join(self.data_path, "Stores.csv")
        features_path = os.path.join(self.data_path, "Features.csv")
        train_path = os.path.join(self.data_path, "Sales.csv") # Sometimes named sales.csv
        
        # Check if files exist
        for p in [stores_path, features_path, train_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"❌ Critical file missing: {p}. Please place it in data/raw/")

        # Load CSVs
        stores = pd.read_csv(stores_path)
        features = pd.read_csv(features_path)
        train = pd.read_csv(train_path)

        # Merge Datasets
        # 1. Merge Train + Stores (on Store)
        # 2. Merge Result + Features (on Store, Date, IsHoliday)
        
        print("🔄 Merging datasets...")
        df_merged = train.merge(stores, on='Store', how='left')
        df_merged = df_merged.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
        
        print(f"✅ Data Loaded Successfully. Shape: {df_merged.shape}")
        return df_merged

if __name__ == "__main__":
    # Test the loader
    loader = RetailDataLoader()
    df = loader.load_data()
    print(df.head())