from src.data_loader import RetailDataLoader
from src.preprocessing import RetailPreprocessor

def test_flow():
    print("🚀 Starting Pipeline Test...\n")

    # 1. Load Data
    try:
        loader = RetailDataLoader(data_path="data/raw")
        raw_df = loader.load_data()
        print(f"📦 Raw Data Loaded. Rows: {len(raw_df)}")
    except Exception as e:
        print(f"❌ Error Loading Data: {e}")
        return

    # 2. Preprocess Data
    try:
        processor = RetailPreprocessor(raw_df)
        clean_df = processor.clean_data()
        clean_df = processor.feature_engineering()
        
        print("\n✨ Preprocessing Output:")
        print("-" * 30)
        print(clean_df.head())
        print("-" * 30)
        print(f"✅ Final Shape: {clean_df.shape}")
        
        # Check specific cleanups
        if 'MarkDown1' in clean_df.columns:
            nulls = clean_df['MarkDown1'].isnull().sum()
            print(f"🔍 MarkDown1 Nulls (Should be 0): {nulls}")
            
    except Exception as e:
        print(f"❌ Error in Preprocessing: {e}")

if __name__ == "__main__":
    test_flow()