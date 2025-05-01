# ARIMA-Stock-Price-Forecast

A Streamlit-based web application that fetches daily stock closing prices via the Alpha Vantage API and uses an ARIMA(p, d, q) time-series model (with an optional drift) to generate short-term forecasts. Interactive parameters let you choose your stock ticker, forecast horizon, and AR/MA orders, and instantly see both a plotted forecast (with 95% confidence intervals) and the underlying data table.

---

## Features

- **Live Data**  
  Retrieves the latest 100 trading days of closing prices for AAPL, MSFT, or AMZN via Alpha Vantage.

- **Auto-Differencing**  
  Automatically determines the optimal degree of differencing (d) via the Augmented Dickey–Fuller test.

- **Customizable AR/MA Orders**  
  User-adjustable `p` (autoregressive) and `q` (moving-average) terms.

- **Drift/Trend Option**  
  Fits a linear time-trend to capture underlying price drift.

- **Interactive Plots**  
  Displays historical prices, forecasted path, and 95% confidence bands using Plotly.

- **Tabular Output**  
  Presents forecasted values in a clean, sortable table.

- **Model Diagnostics**  
  Expandable section to review the full ARIMA model summary.

---

## Installation

1. ### **Clone this repository**  
   ```bash
   git clone https://github.com/your-username/arima-stock-forecast.git
   cd arima-stock-forecast
2. ### (Optional) Create a Virtual Environment
  ```bash
  python -m venv venv
  source venv/bin/activate    # macOS/Linux
  venv\Scripts\activate       # Windows
 ```
3. ### Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
4. ### Input your Alpha Vantage API key


To run the Streamlit app 
  ```bash
  streamlit run personal_project.py
  ```
