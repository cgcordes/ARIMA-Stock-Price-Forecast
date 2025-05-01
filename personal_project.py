import streamlit as st                 # Import Streamlit for building the web app
import pandas as pd                     # Import pandas for data manipulation
from alpha_vantage.timeseries import TimeSeries  # Import Alpha Vantage client for stock data
from statsmodels.tsa.arima.model import ARIMA    # ARIMA model class from statsmodels
from statsmodels.tsa.stattools import adfuller   # Augmented Dickey–Fuller test for stationarity
import plotly.graph_objects as go       # Plotly for interactive plotting
import warnings                         # Warnings module to suppress non-critical alerts

warnings.filterwarnings("ignore")       # Ignore all warnings (e.g. convergence warnings)

# Streamlit page setup
st.set_page_config(page_title="Simple ARIMA Stock Forecast", layout="wide")
                                        # Configure page title and layout width
st.title("📈 ARIMA Stock Price Forecast")  # Main title displayed at top of the app

# Alpha Vantage API setup
API_KEY = "input alpha vantage key here"              # Your Alpha Vantage API key
ts = TimeSeries(key=API_KEY, output_format='pandas')
                                        # Initialize the TimeSeries client to return pandas DataFrames

@st.cache_data(ttl=3600)                # Cache this function’s output for 1 hour
def get_stock_data(ticker):
    """Fetch daily stock data for the given ticker."""
    try:
        data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
                                        # Request the last 100 days of daily data
        data = data.rename(columns={'4. close': 'close'})
                                        # Rename the '4. close' column to 'close'
        data = data[['close']].sort_index()
                                        # Keep only the 'close' column and sort by date ascending
        return data                     # Return the processed DataFrame
    except Exception as e:
        st.error(f"Error fetching data: {e}")  
                                        # Display an error message if API call fails
        return None                     # Return None on error

def is_stationary(series):
    """Return True if series passes the Augmented Dickey–Fuller test (p < 0.05)."""
    return adfuller(series.dropna())[1] < 0.05
                                        # adfuller returns (stat, p-value, ...), we check p-value

def get_d(series):
    """Find the minimal differencing order (0–2) making the series stationary."""
    for d in range(3):                  # Try d = 0, 1, 2
        candidate = series.diff(d).dropna() if d > 0 else series
                                        # If d>0, difference the series; else use original
        if is_stationary(candidate):    # Test for stationarity
            return d                    # Return the first d that works
    return 2                            # Default to d=2 if none are stationary

def arima_forecast(series, steps=30, p=1, q=1):
    """Fit an ARIMA(p,d,q) with linear trend and return the fitted model."""
    d = get_d(series)                  # Automatically find d
    model = ARIMA(series, order=(p, d, q), trend='t').fit()
                                        # Fit ARIMA with trend='t' (linear time trend)
    return model                       # Return the fitted ARIMAResults object

# Sidebar inputs for user to select settings
ticker = st.sidebar.selectbox("Choose a Stock", ["AAPL", "MSFT", "AMZN"])
                                        # Dropdown to pick stock ticker
forecast_days = st.sidebar.slider("Days to Forecast", 1, 60, 30)
                                        # Slider to choose forecast horizon (1–60 days)
st.sidebar.markdown(
    "**p (AR term):** how many past values to regress on  \n"
    "**q (MA term):** size of the moving‐average window"
)                                       # Explanatory text for p and q
p = st.sidebar.slider("AR term (p)", 0, 5, 1)  
                                        # Slider for AR order (0–5)
q = st.sidebar.slider("MA term (q)", 0, 5, 1)  
                                        # Slider for MA order (0–5)

if st.sidebar.button("Run Forecast"):  # Button to trigger data load + forecasting

    with st.spinner("Loading data..."):  
        data = get_stock_data(ticker)  # Fetch stock data

    if data is not None:
        st.subheader(f"{ticker} Stock Closing Prices")
        st.line_chart(data.tail(100))   # Display last 100 days as a line chart

        with st.spinner("Training ARIMA model..."):
            model = arima_forecast(data['close'], steps=forecast_days, p=p, q=q)
                                        # Fit the ARIMA model with chosen parameters

        # ------------------------------------------------------------
        # Plot forecast + confidence intervals
        # ------------------------------------------------------------
        # 1) Build a business-day index for forecast dates
        last_date = data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='B'
        )

        # 2) Generate predictions and 95% confidence intervals
        pred = model.get_forecast(steps=forecast_days)
        pred_mean = pred.predicted_mean
        pred_ci = pred.conf_int()

        # 3) Assign our business-day index to the results
        pred_mean.index = forecast_index
        pred_ci.index = forecast_index

        # 4) Build a DataFrame for display
        forecast_df = pd.DataFrame({'Forecast': pred_mean}, index=forecast_index)

        # 5) Plot historical data, forecast, and intervals
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, y=data['close'],
            mode='lines', name='Historical', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_index, y=forecast_df['Forecast'],
            mode='lines', name='Forecast', line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_index, y=pred_ci.iloc[:, 0],
            mode='lines', name='Lower CI',
            line=dict(dash='dot', color='gray')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_index, y=pred_ci.iloc[:, 1],
            mode='lines', name='Upper CI',
            line=dict(dash='dot', color='gray'),
            fill='tonexty', fillcolor='rgba(128,128,128,0.2)'
        ))
        fig.update_layout(
            title=f"{ticker} Stock Price Forecast",
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)  # Render the Plotly figure

        # 6) Show forecast table beneath the chart
        st.subheader("Forecast Table")
        st.dataframe(forecast_df.round(2))

        # 7) Display model summary in an expander
        with st.expander("Model Summary"):
            st.text(model.summary())

else:
    st.info("Select a stock and click 'Run Forecast' to begin.")  
    # Placeholder message before user runs any forecast
