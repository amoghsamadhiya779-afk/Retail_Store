import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyDetector:
    """
    Detects anomalies in sales data using Statistical and ML methods.
    """
    def __init__(self, df):
        self.df = df

    def detect_outliers_zscore(self, column='Weekly_Sales', threshold=3):
        """
        Detects outliers using Z-Score. Assumes normal distribution.
        """
        print(f"🔍 Running Z-Score Anomaly Detection on {column}...")
        mean = self.df[column].mean()
        std = self.df[column].std()
        
        z_scores = (self.df[column] - mean) / std
        self.df['Z_Score'] = z_scores
        self.df['Is_Anomaly_Z'] = (abs(z_scores) > threshold).astype(int)
        
        n_anomalies = self.df['Is_Anomaly_Z'].sum()
        print(f"   ⚠️ Found {n_anomalies} anomalies (Z-Score > {threshold}).")
        return self.df

    def detect_outliers_iqr(self, column='Weekly_Sales'):
        """
        Detects outliers using Interquartile Range (IQR). Robust to non-normal distributions.
        """
        print(f"🔍 Running IQR Anomaly Detection on {column}...")
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.df['Is_Anomaly_IQR'] = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).astype(int)
        
        n_anomalies = self.df['Is_Anomaly_IQR'].sum()
        print(f"   ⚠️ Found {n_anomalies} anomalies (IQR Method).")
        return self.df

    def detect_isolation_forest(self, column='Weekly_Sales', contamination=0.01):
        """
        Uses Isolation Forest (Unsupervised ML) to detect anomalies.
        """
        print(f"🤖 Running Isolation Forest on {column}...")
        # n_jobs=-1 uses all processors for speed
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        
        # Reshape data for sklearn
        X = self.df[[column]].values
        preds = model.fit_predict(X)
        
        # IsolationForest returns -1 for outliers, 1 for inliers
        # Map: -1 -> 1 (Anomaly), 1 -> 0 (Normal)
        self.df['Is_Anomaly_IF'] = np.where(preds == -1, 1, 0)
        
        n_anomalies = self.df['Is_Anomaly_IF'].sum()
        print(f"   ⚠️ Found {n_anomalies} anomalies (Isolation Forest).")
        return self.df

    def plot_anomalies(self, date_col='Date', sales_col='Weekly_Sales', anomaly_col='Is_Anomaly_IF'):
        """
        Visualizes the anomalies over time.
        """
        plt.figure(figsize=(15, 6))
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
             self.df[date_col] = pd.to_datetime(self.df[date_col])

        # Plot Normal Sales
        normal = self.df[self.df[anomaly_col] == 0]
        plt.scatter(normal[date_col], normal[sales_col], label='Normal', alpha=0.6, s=15, color='#3498db')
        
        # Plot Anomalies
        anomalies = self.df[self.df[anomaly_col] == 1]
        plt.scatter(anomalies[date_col], anomalies[sales_col], label='Anomaly', color='#e74c3c', s=50, marker='x')
        
        plt.title(f'Anomaly Detection: {sales_col} vs Time ({anomaly_col})', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel(sales_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()