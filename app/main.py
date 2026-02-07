import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# --- Setup Path to import src modules ---
# Adds the root project directory to the path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecasting import RetailForecaster
from src.features import FeatureEngineer

# --- Page Config ---
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Spotify-like Glassmorphism UI ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@300;400;500;700&family=Inter:wght@300;400;600&display=swap');
        
        /* --- CSS Variables for Theming --- */
        :root {
            --spotify-green: #1DB954;
            --spotify-black: #121212;
            --spotify-dark-gray: #181818;
            --spotify-light-gray: #282828;
            --spotify-white: #FFFFFF;
            --spotify-text-gray: #B3B3B3;
            
            /* Default Dark Mode Glass */
            --glass-bg: rgba(24, 24, 24, 0.7);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            --card-hover-bg: rgba(40, 40, 40, 0.9);
            --text-main: #FFFFFF;
            --text-secondary: #B3B3B3;
            --bg-color: #121212;
            --sidebar-bg: #000000;
        }

        /* --- Light Mode Overrides --- */
        @media (prefers-color-scheme: light) {
            :root {
                --glass-bg: rgba(255, 255, 255, 0.7);
                --glass-border: rgba(0, 0, 0, 0.08);
                --glass-shadow: 0 4px 30px rgba(0, 0, 0, 0.05);
                --card-hover-bg: rgba(245, 245, 245, 0.9);
                --text-main: #000000;
                --text-secondary: #555555;
                --bg-color: #FFFFFF;
                --sidebar-bg: #F8F9FA;
            }
        }

        /* --- Global App Styling --- */
        .stApp {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            transition: background-color 0.5s ease;
        }

        /* --- Sidebar Styling --- */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--glass-border);
            box-shadow: 5px 0 15px rgba(0,0,0,0.3);
        }
        
        /* Sidebar Navigation Items */
        .stRadio > div[role="radiogroup"] > label {
            background-color: transparent !important;
            border: none;
            padding: 10px 15px;
            border-radius: 8px;
            color: var(--text-secondary) !important;
            font-weight: 500;
            transition: all 0.3s ease;
            margin-bottom: 5px;
        }

        .stRadio > div[role="radiogroup"] > label:hover {
            color: var(--text-main) !important;
            background-color: rgba(29, 185, 84, 0.1) !important;
            transform: translateX(5px);
        }

        /* Active selection styling handled by Streamlit, but we accent it */
        div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
             color: var(--text-main) !important;
             border-left: 3px solid var(--spotify-green);
        }

        /* --- Headings --- */
        h1, h2, h3 {
            color: var(--text-main);
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        h1 span, h2 span, h3 span {
            color: var(--spotify-green);
        }

        /* --- Glassmorphism Card --- */
        .metric-card-container {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: var(--glass-shadow);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }
        
        /* Hover Effect & Animation */
        .metric-card-container:hover {
            transform: translateY(-8px);
            border-color: var(--spotify-green);
            background: var(--card-hover-bg);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        }

        /* Green Glow on Hover */
        .metric-card-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(29, 185, 84, 0.1), transparent);
            transition: left 0.5s;
        }
        
        .metric-card-container:hover::before {
            left: 100%;
        }

        .metric-title {
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 12px;
            font-weight: 600;
        }

        .metric-value {
            color: var(--text-main);
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .metric-delta {
            font-size: 0.95rem;
            color: var(--spotify-green);
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        /* --- Button Styling --- */
        .stButton>button {
            border-radius: 500px !important; /* Pill shape */
            background-color: var(--spotify-green) !important;
            color: #000000 !important;
            font-weight: 700 !important;
            border: none !important;
            padding: 12px 32px !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease !important;
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            background-color: #1ed760 !important; /* Lighter green */
            box-shadow: 0 0 20px rgba(29, 185, 84, 0.6);
        }

        /* --- Input & Select Styles --- */
        div[data-baseweb="select"] > div, .stTextInput > div > div {
            background-color: var(--glass-bg);
            color: var(--text-main);
            border-color: var(--glass-border);
            border-radius: 8px;
        }
        
        /* Plotly Chart Backgrounds */
        .js-plotly-plot .plotly .main-svg {
            background: transparent !important;
        }
        
    </style>
""", unsafe_allow_html=True)

# --- Constants & Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

FILES = {
    "Clean": os.path.join(DATA_DIR, "clean_retail_data.csv"),
    "Anomalies": os.path.join(DATA_DIR, "retail_data_with_anomalies.csv"),
    "Segmented": os.path.join(DATA_DIR, "segmented_retail_data.csv"),
    "MarketBasket": os.path.join(DATA_DIR, "dept_correlation_matrix.csv"),
    "ModelReady": os.path.join(DATA_DIR, "model_ready_data.csv")
}

# --- Helper Functions ---
@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_correlation_matrix(filepath):
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, index_col=0)
    # --- FIX: Ensure Index and Columns are strings to prevent Type Mismatch errors ---
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df

def render_metric_card(title, value, description, icon="📊"):
    st.markdown(f"""
    <div class="metric-card-container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="metric-title">{title}</div>
            <div style="font-size: 1.8rem; color: #1DB954; background: rgba(29, 185, 84, 0.1); border-radius: 50%; padding: 10px; width: 50px; height: 50px; display: flex; justify-content: center; align-items: center;">{icon}</div>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta">
            <span>{description}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🛒 Shopper Spectrum")
    st.markdown("<p style='font-size: 12px; color: #B3B3B3; margin-top: -15px;'>RETAIL ANALYTICS SUITE</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "MENU",
        ["🏠 Home & Overview", 
         "📊 Sales EDA", 
         "🕵️ Anomaly Detection", 
         "👥 Store Segmentation", 
         "🧺 Market Basket Analysis", 
         "📈 Demand Forecasting"],
         label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
        <div style="background: rgba(29, 185, 84, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(29, 185, 84, 0.2);">
            <small style="color: #1DB954; font-weight: bold;">STATUS</small><br>
            <small style="color: var(--text-secondary);">System Online</small><br>
            <small style="color: var(--text-secondary);">v2.0.0 (Spotify UI)</small>
        </div>
    """, unsafe_allow_html=True)

# --- PAGE 1: HOME ---
if page == "🏠 Home & Overview":
    st.title("🏬 Integrated Retail Analytics")
    st.markdown("### <span style='color:#1DB954'>Executive Summary</span>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: var(--glass-bg); padding: 20px; border-radius: 16px; border-left: 5px solid #1DB954; margin-bottom: 20px;">
    Welcome to the Retail Optimization Command Center. This platform integrates advanced machine learning modules to drive decision-making.
    </div>
    """, unsafe_allow_html=True)
    
    # Load basic data for KPIs
    df = load_data(FILES["Clean"])
    
    if df is not None:
        st.markdown("### 🚀 Key Performance Indicators")
        c1, c2, c3, c4 = st.columns(4)
        
        total_sales = df['Weekly_Sales'].sum()
        total_stores = df['Store'].nunique()
        total_depts = df['Dept'].nunique()
        date_range = f"{df['Date'].min().date()} to {df['Date'].max().date()}"
        
        with c1:
            render_metric_card("Total Revenue", f"${total_sales/1e9:.2f}B", "Lifetime Sales Volume", "💰")
        with c2:
            render_metric_card("Active Stores", f"{total_stores}", "Across all regions", "🏢")
        with c3:
            render_metric_card("Departments", f"{total_depts}", "Unique Categories", "📦")
        with c4:
            render_metric_card("Data Period", "2 Years", "2010 - 2012 Analysis", "📅")
            
    else:
        st.error("Data not found. Please run the preprocessing notebooks.")

# --- PAGE 2: EDA ---
elif page == "📊 Sales EDA":
    st.title("📊 Exploratory Data Analysis")
    
    df = load_data(FILES["Clean"])
    if df is not None:
        # 1. Sales over Time
        st.subheader("Total Weekly Sales Trend")
        weekly_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        fig = px.line(weekly_sales, x='Date', y='Weekly_Sales', title='Global Sales Timeline', template="plotly_dark")
        fig.update_traces(line_color='#1DB954', line_width=3)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#B3B3B3")
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Holiday Effect
        st.subheader("Impact of Holidays")
        c1, c2 = st.columns(2)
        with c1:
            avg_holiday = df.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()
            avg_holiday['Type'] = avg_holiday['IsHoliday'].map({0: 'Regular Week', 1: 'Holiday Week'})
            fig_bar = px.bar(avg_holiday, x='Type', y='Weekly_Sales', color='Type', 
                             title="Avg Sales: Holiday vs Regular", template="plotly_dark",
                             color_discrete_sequence=['#535353', '#1DB954'])
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#B3B3B3")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with c2:
            st.markdown("""
            <div class="metric-card-container">
            <h4 style="color:#1DB954">Observation</h4>
            <p>Holiday weeks often show a significant spike in sales volume.
            However, the 'dip' immediately after holidays is also a crucial pattern to watch.</p>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. Correlation Heatmap
        st.subheader("Economic Indicator Correlations")
        corr_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        # Filter for existing columns
        corr_cols = [c for c in corr_cols if c in df.columns]
        
        if corr_cols:
            corr = df[corr_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Greens", 
                                 title="Feature Correlations", template="plotly_dark")
            fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#B3B3B3")
            st.plotly_chart(fig_corr, use_container_width=True)

# --- PAGE 3: ANOMALY DETECTION ---
elif page == "🕵️ Anomaly Detection":
    st.title("🕵️ Anomaly Detection")
    st.markdown("Identifying unusual sales patterns using **Isolation Forest**.")
    
    df_anom = load_data(FILES["Anomalies"])
    
    if df_anom is not None:
        # Filter options
        store_select = st.selectbox("Select Store to Inspect:", sorted(df_anom['Store'].unique()))
        subset = df_anom[df_anom['Store'] == store_select]
        
        # Aggregate to store level for cleaner plot
        # Check if Is_Anomaly_IF exists, otherwise default to 0
        if 'Is_Anomaly_IF' not in subset.columns:
             subset['Is_Anomaly_IF'] = 0
             
        store_weekly = subset.groupby(['Date', 'Is_Anomaly_IF'])['Weekly_Sales'].sum().reset_index()
        
        st.subheader(f"Sales Anomalies for Store {store_select}")
        
        fig = px.scatter(
            store_weekly, 
            x='Date', 
            y='Weekly_Sales', 
            color=store_weekly['Is_Anomaly_IF'].astype(str),
            color_discrete_map={'0': '#1DB954', '1': '#E91429'},
            title="Normal Sales (Green) vs Anomalies (Red)",
            hover_data=['Weekly_Sales'],
            template="plotly_dark"
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#B3B3B3")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Anomalies often correspond to specific events like **Black Friday** or **Christmas**, but can also indicate data errors or local events.")

# --- PAGE 4: SEGMENTATION ---
elif page == "👥 Store Segmentation":
    st.title("👥 Customer & Store Segmentation")
    st.markdown("Clustering stores based on Volume, Size, and Markdown Sensitivity.")
    
    df_seg = load_data(FILES["Segmented"])
    
    if df_seg is not None:
        # --- FIX: Ensure column naming consistency ---
        if 'Size' in df_seg.columns and 'Store_Size' not in df_seg.columns:
            df_seg = df_seg.rename(columns={'Size': 'Store_Size'})

        # Get one row per store (since it's time series data joined with clusters)
        # Group by Store & Cluster, then mean.
        # We assume Total_MarkDown exists (created in Feature Engineering)
        
        cols_to_agg = ['Weekly_Sales', 'Store_Size']
        if 'Total_MarkDown' in df_seg.columns:
            cols_to_agg.append('Total_MarkDown')
            
        store_profile = df_seg.groupby(['Store', 'Cluster'])[cols_to_agg].mean().reset_index()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Store Clusters Visualization")
            
            # Dynamic Axis Selection
            avail_metrics = cols_to_agg
            x_axis = st.selectbox("X Axis", avail_metrics, index=min(1, len(avail_metrics)-1))
            y_axis = st.selectbox("Y Axis", avail_metrics, index=0)
            
            fig = px.scatter(
                store_profile, 
                x=x_axis, 
                y=y_axis, 
                color='Cluster', 
                size='Weekly_Sales',
                hover_name='Store',
                title=f"Clusters: {y_axis} vs {x_axis}",
                template="plotly_dark",
                color_continuous_scale="Greens"
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#B3B3B3")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Cluster Profile")
            
            # Render dataframe with some styling
            st.dataframe(store_profile.groupby('Cluster').mean().round(2).T, use_container_width=True)
            
            st.markdown("""
            <div style="background: var(--glass-bg); padding: 15px; border-radius: 10px; border: 1px solid var(--glass-border);">
            <small style="color: #1DB954;"><b>Interpretation:</b></small><br>
            <ul style="color: var(--text-secondary); margin-top: 5px; padding-left: 20px;">
                <li><b>X/Y Axes:</b> Toggle to see Size vs Sales.</li>
                <li><b>Color:</b> Performance groups.</li>
                <li><b>Size:</b> Weekly Sales Volume.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

# --- PAGE 5: MARKET BASKET ---
elif page == "🧺 Market Basket Analysis":
    st.title("🧺 Market Basket Analysis")
    st.markdown("Analyzing Department Affinity to optimize cross-selling.")
    
    corr_matrix = load_correlation_matrix(FILES["MarketBasket"])
    
    if corr_matrix is not None:
        st.subheader("Department Correlation Heatmap")
        
        # Filter for top interactive view
        all_depts = corr_matrix.columns.tolist()
        
        # Default selection (Top 10)
        default_sel = all_depts[:10] if len(all_depts) > 10 else all_depts
        selected_depts = st.multiselect("Select Departments to Compare:", all_depts, default=default_sel)
        
        if selected_depts:
            # Filter matrix
            subset_corr = corr_matrix.loc[selected_depts, selected_depts]
            fig = px.imshow(
                subset_corr, 
                text_auto=False, 
                color_continuous_scale="Greens",
                title="Department Affinity Strength",
                template="plotly_dark"
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#B3B3B3")
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("### 💡 Strategy")
        c1, c2 = st.columns(2)
        with c1:
            st.success("**High Correlation?** Bundle these items or place them close together.")
        with c2:
            st.warning("**Negative Correlation?** Likely seasonal substitutes.")
    else:
        st.error("Market Basket Data not found. Run Notebook 06.")

# --- PAGE 6: FORECASTING ---
elif page == "📈 Demand Forecasting":
    st.title("📈 AI Demand Forecasting")
    st.markdown("Predict future sales using **Facebook Prophet**.")
    
    df_model = load_data(FILES["ModelReady"])
    
    if df_model is not None:
        # Styled container for inputs
        st.markdown('<div class="metric-card-container">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            store_id = st.selectbox("Select Store", sorted(df_model['Store'].unique()))
        with c2:
            # Filter depts for this store
            avail_depts = sorted(df_model[df_model['Store'] == store_id]['Dept'].unique())
            if not avail_depts:
                st.warning("No depts found.")
                dept_id = None
            else:
                dept_id = st.selectbox("Select Department", avail_depts)
        with c3:
            horizon = st.slider("Forecast Horizon (Weeks)", 4, 24, 12)
        st.markdown('</div>', unsafe_allow_html=True)
            
        if st.button("🚀 Generate Forecast") and dept_id is not None:
            with st.spinner("Training Prophet Model... (This takes a few seconds)"):
                try:
                    # Initialize Forecaster
                    forecaster = RetailForecaster(df_model)
                    
                    # Prepare Data
                    subset = forecaster.prepare_data(store_id, dept_id)
                    train_df = subset # Use all data for training in the app
                    
                    # Train
                    regressors = ['CPI', 'Unemployment', 'Fuel_Price', 'IsHoliday']
                    # Check if regressors exist in data
                    regressors = [r for r in regressors if r in train_df.columns]
                    
                    model = forecaster.train_model(train_df, regressors=regressors)
                    
                    # Create Future Regressors (Mocking future economic stability for demo)
                    last_row = subset.iloc[-1]
                    future_dates = pd.date_range(start=last_row['ds'], periods=horizon+1, freq='W')[1:]
                    
                    future_data = {'ds': future_dates}
                    for reg in regressors:
                        if reg == 'IsHoliday':
                            future_data[reg] = 0 # Assume no holidays for simplicity in demo
                        else:
                            future_data[reg] = last_row[reg] # Assume constant economy
                            
                    future_regressors = pd.DataFrame(future_data)
                    
                    # Predict
                    forecast = forecaster.predict(periods=horizon, future_regressors_df=future_regressors)
                    
                    # Visualize
                    st.subheader(f"Sales Forecast: Store {store_id} - Dept {dept_id}")
                    
                    # Plotly viz for forecast
                    fig = go.Figure()
                    
                    # Actuals
                    fig.add_trace(go.Scatter(x=subset['ds'], y=subset['y'], name='Actual Sales', line=dict(color='#FFFFFF', width=2)))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted', line=dict(color='#1DB954', width=3)))
                    
                    # Confidence Interval
                    fig.add_trace(go.Scatter(
                        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(29, 185, 84, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title="Forecast vs History", 
                        xaxis_title="Date", 
                        yaxis_title="Sales",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#B3B3B3"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show Data
                    st.write("Forecast Data:", forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon))
                    
                except Exception as e:
                    st.error(f"Forecasting Error: {e}")