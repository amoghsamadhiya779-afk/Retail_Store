import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class MarketBasketAnalyzer:
    """
    Analyzes affinities between Departments.
    Since we don't have individual items, we treat 'Departments' as items 
    and (Store, Date) as the 'Transaction'.
    """
    def __init__(self, df):
        self.df = df
        self.pivot_table = None
        self.binary_table = None

    def create_basket(self):
        """
        Pivots data so rows = (Store, Date) and Columns = Dept.
        Values = Weekly_Sales.
        """
        print("🧺 Creating Basket (Pivot Table)...")
        # We focus on positive sales only
        valid_sales = self.df[self.df['Weekly_Sales'] > 0]
        
        # Pivot: Index=(Store, Date), Col=Dept, Val=Weekly_Sales
        self.pivot_table = valid_sales.pivot_table(
            index=['Store', 'Date'], 
            columns='Dept', 
            values='Weekly_Sales', 
            fill_value=0
        )
        
        # Binary table (1 if Dept sold anything, 0 otherwise) for Co-occurrence
        self.binary_table = self.pivot_table.applymap(lambda x: 1 if x > 0 else 0)
        
        print(f"   ✅ Basket Created. Transactions: {self.pivot_table.shape[0]}, Items (Depts): {self.pivot_table.shape[1]}")
        return self.pivot_table

    def analyze_correlation(self, min_correlation=0.5):
        """
        Calculates Pearson correlation between Department Sales.
        Returns pairs with correlation > min_correlation.
        """
        print("🔗 Calculating Sales Correlation...")
        corr_matrix = self.pivot_table.corr()
        
        # Unstack to get pairs
        corr_pairs = corr_matrix.unstack()
        sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
        
        # Filter self-correlation and low correlation
        strong_pairs = sorted_pairs[
            (sorted_pairs > min_correlation) & (sorted_pairs < 1.0)
        ]
        
        print(f"   Found {len(strong_pairs)//2} strong correlations.")
        return corr_matrix, strong_pairs

    def plot_affinity_heatmap(self, corr_matrix, top_n=20):
        """
        Plots a heatmap of the top N departments.
        """
        # Select top N departments by Total Volume to make heatmap readable
        top_depts = self.df.groupby('Dept')['Weekly_Sales'].sum().nlargest(top_n).index
        subset_corr = corr_matrix.loc[top_depts, top_depts]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(subset_corr, cmap='coolwarm', center=0, annot=False)
        plt.title(f'Department Sales Affinity (Top {top_n} Depts)')
        plt.show()