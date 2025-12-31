#  PredictPC: Intelligent Hardware Price Forecasting

**PredictPC** is a machine learning-powered price intelligence tool designed to forecast the prices of PC components (GPUs, CPUs) using historical market data.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![ML](https://img.shields.io/badge/Model-Prophet%20%7C%20GBM-green)

##  Project Overview
Buying PC parts at the right time is difficult due to volatile market prices. **PredictPC** solves this by:
1.  **Tracking** daily prices via the Keepa API.
2.  **Analyzing** trends using statistical and ML models.
3.  **Forecasting** prices 7 days into the future to recommend the optimal purchase window.

##  Architecture
The system follows a modular pipeline:
*   **Data Ingestion**: Fetches historical price and rank data from Keepa.
*   **Storage**: Efficiently stores time-series data in a local SQLite database (data/predictpc.db).
*   **Modeling**:
    *   **Prophet**: Captures seasonality and trends.
    *   **GBM (Gradient Boosting)**: utilized for complex non-linear patterns.
*   **Visualization**: A responsive **Streamlit** dashboard for real-time interaction.

##  Features
-   **Multi-Category Tracking**: Monitors GPUs and CPUs (customizable in src/config.py).
-   **Interactive Dashboard**: Zoomable price history charts with forecast overlays.
-   **Configurable Horizon**: Modify prediction windows and history lookback.
-   **Dark Mode UI**: Modern, aesthetic interface using custom CSS.

##  Installation & Setup

1.  **Clone the repository**
    `ash
    git clone https://github.com/yourusername/PredictPC.git
    cd PredictPC/predictpc
    `

2.  **Create a Virtual Environment**
    `ash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\Activate.ps1
    # Linux/Mac
    source .venv/bin/activate
    `

3.  **Install Dependencies**
    `ash
    pip install -r requirements.txt
    `

4.  **Environment Configuration**
    Create a .env file in the root predictpc folder:
    `env
    KEEPA_API_KEY=your_keepa_api_key_here
    `

5.  **Run the Dashboard**
    `ash
    streamlit run src/dashboard/app.py
    `

##  Technologies Used
-   **Core**: Python, Pandas, NumPy
-   **ML**: Facebook Prophet, Scikit-learn, LightGBM/XGBoost
-   **Viz**: Plotly, Streamlit
-   **Data**: Keepa API, SQLAlchemy

---
*Created by [Your Name] - Machine Learning Engineer*
