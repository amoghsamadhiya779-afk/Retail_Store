import os
import sys

def create_structure():
    """
    Generates a professional folder structure for the Retail Analytics Capstone Project.
    """
    project_name = "Retail_Analytics_Optimization"
    
    # Define the structure
    structure = {
        "data": [
            "raw",            # Original datasets (Stores, Features, Sales)
            "processed",      # Cleaned/Merged data ready for ML
            "external"        # CPI, Unemployment, Fuel Price data
        ],
        "notebooks": [
            "01_Data_Cleaning_and_EDA.ipynb",       # Handling Markdowns, Nulls, Merging
            "02_Hypothesis_Testing.ipynb",          # Statistical tests on Holiday effects
            "03_Anomaly_Detection.ipynb",           # Isolation Forest / Time-series outliers
            "04_Feature_Engineering.ipynb",         # Lag features, Rolling windows
            "05_Customer_Segmentation.ipynb",       # Clustering Stores/Depts
            "06_Market_Basket_Analysis.ipynb",      # Association Rules
            "07_Demand_Forecasting.ipynb",          # Time-Series (Prophet/ARIMA/LSTM)
            "08_Model_Evaluation.ipynb"             # Final comparison
        ],
        "src": [
            "__init__.py",
            "data_loader.py",       # Script to load and merge datasets
            "preprocessing.py",     # Cleaning logic (MarkDown imputation)
            "features.py",          # Feature engineering logic
            "models.py",            # Forecasting & Clustering classes
            "utils.py"              # Plotting and helper functions
        ],
        "app": [
            "main.py",              # Streamlit Dashboard Entry Point
            "pages"                 # Sub-pages for the app
        ],
        "models": [
            "saved_models"          # Folder for .pkl, .h5 files
        ],
        "reports": [
            "figures",              # Saved plots for the report
            "final_report"          # Place for the PDF/PPTX deliverables
        ]
    }

    print(f"🚀 Initializing Project Structure for: {project_name}")

    # Create directories and files
    for folder, contents in structure.items():
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created: {folder}/")
        
        for item in contents:
            path = os.path.join(folder, item)
            
            # If it doesn't have an extension, treat as folder
            if "." not in item:
                os.makedirs(path, exist_ok=True)
                print(f"  📂 Created subfolder: {item}/")
            else:
                # It's a file
                if not os.path.exists(path):
                    with open(path, 'w') as f:
                        pass # Create empty file
                    print(f"  📄 Created file: {item}")

    # Create Root Level Files
    root_files = {
        "README.md": "# Integrated Retail Analytics\nDocumentation goes here.",
        "requirements.txt": "pandas\nnumpy\nscikit-learn\nmatplotlib\nseaborn\nstatsmodels\nprophet\nstreamlit\nxgboost",
        ".gitignore": "venv/\n__pycache__/\n.ipynb_checkpoints/\ndata/\nmodels/saved_models/\n.env\n*.DS_Store"
    }

    for filename, content in root_files.items():
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(content)
            print(f"📄 Created root file: {filename}")

    print("\n✅ Project Structure Created Successfully!")
    print("👉 ACTION REQUIRED: Place your dataset files in 'data/raw/'")

if __name__ == "__main__":
    create_structure()