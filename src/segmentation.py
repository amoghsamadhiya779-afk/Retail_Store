import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class StoreSegmenter:
    """
    Handles clustering of Stores based on Sales, Markdowns, and Size.
    """
    def __init__(self, df):
        self.df = df.copy()
        self.store_profile = None
        self.scaled_data = None
        self.kmeans_model = None

    def aggregate_store_metrics(self):
        """
        Aggregates the time-series data into a single profile per Store.
        """
        print("📊 Aggregating metrics by Store...")
        
        # Base aggregation
        agg_funcs = {
            'Weekly_Sales': ['sum', 'mean', 'std'],
            'Total_MarkDown': 'sum',
            'CPI': 'mean',
            'Unemployment': 'mean'
        }
        
        # Dynamically add static features if they exist
        if 'Size' in self.df.columns:
            agg_funcs['Size'] = 'first'
        if 'Type_Encoded' in self.df.columns:
            agg_funcs['Type_Encoded'] = 'first'
            
        # Group by Store
        self.store_profile = self.df.groupby('Store').agg(agg_funcs)
        
        # Robust Column Flattening (Handles MultiIndex correctly)
        new_cols = []
        for col in self.store_profile.columns.values:
            if isinstance(col, tuple):
                # Join tuple ('Weekly_Sales', 'sum') -> 'Weekly_Sales_sum'
                new_cols.append('_'.join(col).strip())
            else:
                new_cols.append(col)
        
        self.store_profile.columns = new_cols
        
        # Rename for readability
        rename_map = {
            'Weekly_Sales_sum': 'Lifetime_Sales',
            'Weekly_Sales_mean': 'Avg_Weekly_Sales',
            'Weekly_Sales_std': 'Sales_Volatility',
            'Total_MarkDown_sum': 'Total_Markdown_Volume',
            'Size_first': 'Store_Size',
            'Type_Encoded_first': 'Store_Type'
        }
        self.store_profile = self.store_profile.rename(columns=rename_map)
        
        # Fill NaN (e.g., if std dev is NaN due to single record)
        self.store_profile = self.store_profile.fillna(0)
        
        print(f"   ✅ Aggregation Complete. Shape: {self.store_profile.shape}")
        return self.store_profile

    def preprocess_data(self):
        """
        Scales the data for K-Means (StandardScaler).
        """
        if self.store_profile is None:
            raise ValueError("Store Profile is empty. Run aggregate_store_metrics() first.")

        scaler = StandardScaler()
        # Features to cluster on
        features = ['Avg_Weekly_Sales', 'Sales_Volatility', 'Total_Markdown_Volume']
        
        # Add Store_Size if available
        if 'Store_Size' in self.store_profile.columns:
            features.append('Store_Size')
            
        # Check for missing columns
        missing = [f for f in features if f not in self.store_profile.columns]
        if missing:
            print(f"   ⚠️ Warning: Missing features for clustering: {missing}")
            features = [f for f in features if f in self.store_profile.columns]

        # Prepare data
        data_to_scale = self.store_profile[features]
        
        # Handle Infinite values
        data_to_scale = data_to_scale.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        self.scaled_data = scaler.fit_transform(data_to_scale)
        return self.scaled_data

    def find_optimal_k(self, max_k=10):
        """
        Plots the Elbow Curve to help choose k.
        """
        print("📉 Calculating Elbow Curve...")
        inertias = []
        k_range = range(1, min(max_k+1, len(self.store_profile))) # Safe range
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            inertias.append(kmeans.inertia_)
            
        plt.figure(figsize=(8, 4))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (WCSS)')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.show()

    def train_clustering(self, k=3):
        """
        Fits K-Means with the chosen k.
        """
        print(f"🤖 Training K-Means with k={k}...")
        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(self.scaled_data)
        
        self.store_profile['Cluster'] = clusters
        return self.store_profile

    def visualize_pca(self):
        """
        Uses PCA to visualize the multidimensional clusters in 2D.
        """
        if self.scaled_data is None:
             print("❌ No scaled data found. Cannot visualize.")
             return

        pca = PCA(n_components=2)
        components = pca.fit_transform(self.scaled_data)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=components[:,0], 
            y=components[:,1], 
            hue=self.store_profile['Cluster'], 
            palette='viridis', 
            s=100, 
            alpha=0.8
        )
        plt.title('Store Clusters (PCA Visualization)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.show()