🛒 Shopper Spectrum: Integrated Retail Analytics

📖 Executive Summary

Shopper Spectrum is an end-to-end Machine Learning solution designed to optimize retail store performance. By integrating internal sales data with external economic indicators (CPI, Unemployment, Fuel Prices), the system transforms raw transaction data into actionable strategic insights.

The platform addresses three critical business challenges:

Strategic Segmentation: Grouping stores by performance and behavior (Clustering).

Cross-Selling: Identifying product affinities to optimize layout and bundles (Market Basket).

Future Demand: Predicting weekly sales with high accuracy using additive regression models (Forecasting).

The solution is delivered via a modern, glassmorphism-styled Streamlit Dashboard inspired by Web3 aesthetics.

🏗️ Project Architecture

The project follows a modular, production-ready structure:

Retail_Analytics_Optimization/
├── app/
│   └── main.py               # Streamlit Dashboard Entry Point (Spotify-like UI)
├── data/
│   ├── raw/                  # Original datasets (Stores, Features, Sales)
│   └── processed/            # Cleaned & Engineered datasets
├── models/                   # Serialized ML models (.pkl files)
├── notebook/
│   ├── 01_Data_Cleaning...   # Preprocessing pipeline
│   ├── 02_Hypothesis...      # Statistical validation
│   ├── 03_Anomaly_Detect...  # Isolation Forest logic
│   ├── 04_Feature_Eng...     # Time-series feature generation
│   ├── 05_Segmentation...    # K-Means Clustering
│   ├── 06_Market_Basket...   # Affinity Analysis
│   ├── 07_Forecasting...     # Prophet Modeling
│   └── 08_Evaluation...      # Cross-Validation & Metrics
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data ingestion handler
│   ├── preprocessing.py      # Cleaning & Feature Engineering pipeline
│   ├── anomaly_detection.py  # Outlier detection logic
│   ├── segmentation.py       # Clustering logic
│   ├── market_basket.py      # Association rule logic
│   ├── features.py           # Advanced feature engineering
│   └── forecasting.py        # Prophet wrapper class
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation


🚀 Key Features & Modules

1. 📊 Exploratory Data Analysis (EDA)

Time-Series Decomposition: Visualizing global sales trends and seasonality.

Holiday Impact: Quantifying the sales spike during major events (Super Bowl, Christmas).

Economic Correlation: Heatmaps showing the relationship between Sales, Fuel Price, CPI, and Unemployment.

2. 🕵️ Anomaly Detection

Algorithm: Isolation Forest (Unsupervised Learning).

Goal: Detect unusual sales patterns that don't match seasonal trends.

Insight: Separates true data errors from significant local events.

3. 👥 Store Segmentation

Algorithm: K-Means Clustering on aggregated Store Profiles.

Features: Lifetime Sales, Volatility, Store Size, Markdown Sensitivity.

Output: Clusters stores into performance tiers (e.g., "High Volume Flagships" vs. "Markdown Dependent").

4. basket Market Basket Analysis

Method: Department-level correlation and co-occurrence analysis.

Strategy: Identifies complementary departments to optimize store layout and cross-promotional bundles.

5. 📈 AI Demand Forecasting

Model: Facebook Prophet.

Capabilities:

Handles multi-seasonal patterns (Weekly + Yearly).

Incorporates external regressors (Economy & Holidays).

Provides 12-24 week future horizons with confidence intervals.

⚙️ Installation & Setup

Prerequisites

Python 3.8 or higher.

Git.

1. Clone the Repository

git clone [https://github.com/amoghsamadhiya779-afk/shopper-spectrum.git](https://github.com/amoghsamadhiya779-afk/shopper-spectrum.git)
cd shopper-spectrum


2. Set Up Environment

It is recommended to use a virtual environment.

# Create venv
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Data Setup

Place your raw datasets (Sales.csv, Stores.csv, Features.csv) inside the data/raw/ folder.
(Note: If using the Walmart Recruiting dataset, rename train.csv to Sales.csv or rely on the data loader's fallback logic).

🖥️ Usage Guide

Running the Analysis Pipeline

To generate the models and processed data from scratch, run the notebooks in sequential order (01 through 08).

Launching the Dashboard

To start the interactive web application:

streamlit run app/main.py
🎨 UI Design System
The dashboard features a custom Dark Mode / Glassmorphism design:

Palette: Deep Black (#121212) backgrounds with Neon Green (#1DB954) accents.

Interactivity: Hover-responsive cards, animated charts, and glass-effect overlays.

Responsiveness: Adapts to system theme settings (Light/Dark mode supported).

📈 Future Roadmap
Deep Learning: Implementing LSTM/GRU for potentially higher forecasting accuracy on complex patterns.

Deployment: Dockerizing the application for cloud deployment (AWS/Azure).

Live Data: Integrating a SQL database for real-time transaction ingestion.

👥 Contributors
[Amogh Samadhiya] - Project Lead & Developer

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.