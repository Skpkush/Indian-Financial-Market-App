import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import base64
from io import StringIO
import json
import requests
import ta  # Technical analysis library
import calendar
from scipy.stats import norm, skew, kurtosis
from nsetools import Nse

# Set page configuration
st.set_page_config(
    page_title="Financial Data",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Indian Financial Market APP")
st.markdown("Access and analyze financial data for the Nifty 50 stocks from the Indian Stock Market.")

# Function to download dataframe as CSV
def get_csv_download_link(df, filename="stock_data.csv", text="Download CSV"):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Nifty 50 stocks (as of March 2025)
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_nifty50_stocks():
    # This is the list of Nifty 50 stocks with their Yahoo Finance symbols
    nifty_50_symbols = {
        'RELIANCE.NS': 'Reliance Industries Ltd.',
        'TCS.NS': 'Tata Consultancy Services Ltd.',
        'HDFCBANK.NS': 'HDFC Bank Ltd.',
        'INFY.NS': 'Infosys Ltd.',
        'ICICIBANK.NS': 'ICICI Bank Ltd.',
        'HINDUNILVR.NS': 'Hindustan Unilever Ltd.',
        'ITC.NS': 'ITC Ltd.',
        'SBIN.NS': 'State Bank of India',
        'BHARTIARTL.NS': 'Bharti Airtel Ltd.',
        'KOTAKBANK.NS': 'Kotak Mahindra Bank Ltd.',
        'LT.NS': 'Larsen & Toubro Ltd.',
        'AXISBANK.NS': 'Axis Bank Ltd.',
        'ASIANPAINT.NS': 'Asian Paints Ltd.',
        'MARUTI.NS': 'Maruti Suzuki India Ltd.',
        'BAJFINANCE.NS': 'Bajaj Finance Ltd.',
        'HCLTECH.NS': 'HCL Technologies Ltd.',
        'SUNPHARMA.NS': 'Sun Pharmaceutical Industries Ltd.',
        'TITAN.NS': 'Titan Company Ltd.',
        'TATAMOTORS.NS': 'Tata Motors Ltd.',
        'WIPRO.NS': 'Wipro Ltd.',
        'NTPC.NS': 'NTPC Ltd.',
        'POWERGRID.NS': 'Power Grid Corporation of India Ltd.',
        'ULTRACEMCO.NS': 'UltraTech Cement Ltd.',
        'M&M.NS': 'Mahindra & Mahindra Ltd.',
        'ONGC.NS': 'Oil & Natural Gas Corporation Ltd.',
        'BAJAJFINSV.NS': 'Bajaj Finserv Ltd.',
        'TATASTEEL.NS': 'Tata Steel Ltd.',
        'NESTLEIND.NS': 'Nestle India Ltd.',
        'TECHM.NS': 'Tech Mahindra Ltd.',
        'COALINDIA.NS': 'Coal India Ltd.',
        'ADANIPORTS.NS': 'Adani Ports and Special Economic Zone Ltd.',
        'GRASIM.NS': 'Grasim Industries Ltd.',
        'HDFCLIFE.NS': 'HDFC Life Insurance Company Ltd.',
        'INDUSINDBK.NS': 'IndusInd Bank Ltd.',
        'HINDALCO.NS': 'Hindalco Industries Ltd.',
        'BAJAJ-AUTO.NS': 'Bajaj Auto Ltd.',
        'SBILIFE.NS': 'SBI Life Insurance Company Ltd.',
        'EICHERMOT.NS': 'Eicher Motors Ltd.',
        'DIVISLAB.NS': 'Divi\'s Laboratories Ltd.',
        'DRREDDY.NS': 'Dr. Reddy\'s Laboratories Ltd.',
        'BRITANNIA.NS': 'Britannia Industries Ltd.',
        'JSWSTEEL.NS': 'JSW Steel Ltd.',
        'CIPLA.NS': 'Cipla Ltd.',
        'HEROMOTOCO.NS': 'Hero MotoCorp Ltd.',
        'APOLLOHOSP.NS': 'Apollo Hospitals Enterprise Ltd.',
        'UPL.NS': 'UPL Ltd.',
        'SHREECEM.NS': 'Shree Cement Ltd.',
        'BPCL.NS': 'Bharat Petroleum Corporation Ltd.',
        'IOC.NS': 'Indian Oil Corporation Ltd.',
        'TATACONSUM.NS': 'Tata Consumer Products Ltd.'
    }
    
    return nifty_50_symbols

# Function to get basic info for multiple stocks
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_multiple_stock_info(tickers):
    result = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            # Extract only essential info to keep the result size manageable
            essential_info = {
                'symbol': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'price': info.get('regularMarketPrice', 'N/A'),
                'change': info.get('regularMarketChangePercent', 'N/A'),
                'marketCap': info.get('marketCap', 'N/A'),
                'pe': info.get('trailingPE', 'N/A'),
                'eps': info.get('trailingEps', 'N/A'),
                'dividend': info.get('dividendYield', 'N/A'),
            }
            result[ticker] = essential_info
        except Exception as e:
            st.error(f"Error retrieving data for {ticker}: {e}")
    
    return result

# Function to get stock data from Yahoo Finance
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            return None, None
        return df, stock.info
    except Exception as e:
        st.error(f"Error retrieving data: {e}")
        return None, None

# Function to add technical indicators to stock data
def add_technical_indicators(df):
    if df is None or df.empty:
        return df
    
    # Make a copy to avoid modifying the original dataframe
    df_with_indicators = df.copy()
    
    # Moving Averages
    df_with_indicators['MA5'] = ta.trend.sma_indicator(df_with_indicators['Close'], window=5)
    df_with_indicators['MA20'] = ta.trend.sma_indicator(df_with_indicators['Close'], window=20)
    df_with_indicators['MA50'] = ta.trend.sma_indicator(df_with_indicators['Close'], window=50)
    df_with_indicators['MA200'] = ta.trend.sma_indicator(df_with_indicators['Close'], window=200)
    
    # Exponential Moving Averages
    df_with_indicators['EMA9'] = ta.trend.ema_indicator(df_with_indicators['Close'], window=9)
    df_with_indicators['EMA21'] = ta.trend.ema_indicator(df_with_indicators['Close'], window=21)
    
    # MACD
    macd = ta.trend.MACD(df_with_indicators['Close'])
    df_with_indicators['MACD'] = macd.macd()
    df_with_indicators['MACD_Signal'] = macd.macd_signal()
    df_with_indicators['MACD_Hist'] = macd.macd_diff()
    
    # RSI (Relative Strength Index)
    df_with_indicators['RSI'] = ta.momentum.RSIIndicator(df_with_indicators['Close']).rsi()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df_with_indicators['Close'])
    df_with_indicators['BB_Upper'] = bollinger.bollinger_hband()
    df_with_indicators['BB_Lower'] = bollinger.bollinger_lband()
    df_with_indicators['BB_Middle'] = bollinger.bollinger_mavg()
    
    # Average True Range (ATR) - Volatility indicator
    df_with_indicators['ATR'] = ta.volatility.AverageTrueRange(
        df_with_indicators['High'], 
        df_with_indicators['Low'], 
        df_with_indicators['Close']
    ).average_true_range()
    
    # Calculate daily returns
    df_with_indicators['Daily_Return'] = df_with_indicators['Close'].pct_change() * 100
    
    # Calculate volatility (20-day rolling standard deviation of returns)
    df_with_indicators['Volatility'] = df_with_indicators['Daily_Return'].rolling(window=20).std()
    
    # Add volume moving average
    df_with_indicators['Volume_MA20'] = ta.trend.sma_indicator(df_with_indicators['Volume'], window=20)
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        df_with_indicators['High'], 
        df_with_indicators['Low'], 
        df_with_indicators['Close']
    )
    df_with_indicators['Stoch_K'] = stoch.stoch()
    df_with_indicators['Stoch_D'] = stoch.stoch_signal()
    
    return df_with_indicators

# Calculate support and resistance levels
def calculate_support_resistance(df, window=10):
    if df is None or df.empty or len(df) < window*2:
        return [], []
    
    df = df.copy()
    
    # Find local minima and maxima
    df['min'] = df['Low'].rolling(window=window, center=True).min()
    df['max'] = df['High'].rolling(window=window, center=True).max()
    
    # Identify support levels (local minima)
    supports = []
    for i in range(window, len(df) - window):
        if df['Low'].iloc[i] == df['min'].iloc[i] and df['Low'].iloc[i] != df['Low'].iloc[i-1]:
            supports.append((df.index[i], df['Low'].iloc[i]))
    
    # Identify resistance levels (local maxima)
    resistances = []
    for i in range(window, len(df) - window):
        if df['High'].iloc[i] == df['max'].iloc[i] and df['High'].iloc[i] != df['High'].iloc[i-1]:
            resistances.append((df.index[i], df['High'].iloc[i]))
    
    # Get recent supports and resistances (last 5 of each)
    supports = sorted(supports, key=lambda x: x[0], reverse=True)[:5]
    resistances = sorted(resistances, key=lambda x: x[0], reverse=True)[:5]
    
    return supports, resistances

# Perform correlation analysis with other stocks or indices
def get_correlation_analysis(ticker, benchmark_tickers, start_date, end_date):
    """
    Calculate correlations between a stock and benchmark indices/stocks
    """
    all_tickers = [ticker] + benchmark_tickers
    correlation_data = pd.DataFrame()
    
    # Get data for all tickers
    for t in all_tickers:
        try:
            df = yf.download(t, start=start_date, end=end_date, progress=False)['Close']
            if not df.empty:
                correlation_data[t] = df
        except Exception as e:
            st.warning(f"Could not fetch data for {t}: {e}")
    
    # Calculate daily returns for correlation
    returns_data = correlation_data.pct_change().dropna()
    
    # Calculate correlation matrix
    correlation_matrix = returns_data.corr()
    
    # Calculate beta (against the first benchmark which is typically the market index)
    if len(benchmark_tickers) > 0 and benchmark_tickers[0] in returns_data.columns and ticker in returns_data.columns:
        benchmark_returns = returns_data[benchmark_tickers[0]]
        stock_returns = returns_data[ticker]
        
        # Calculate covariance and variance
        covariance = stock_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        
        # Calculate beta
        beta = covariance / benchmark_variance
    else:
        beta = None
    
    return correlation_matrix, beta

# Initialize session state for temp ticker
if 'temp_ticker' not in st.session_state:
    st.session_state.temp_ticker = None


# Add this to your Streamlit app where you want the stock selection to happen
def stock_selector():
    # Get the dictionary of Nifty 50 stocks
    nifty_stocks = get_nifty50_stocks()
    
    # Create a mapping from company name to ticker symbol
    name_to_symbol = {name: symbol for symbol, name in nifty_stocks.items()}
    
    # Create the dropdown with company names
    selected_company = st.selectbox(
        "Select a Nifty 50 company:",
        options=list(nifty_stocks.values())
    )
    
    # Get the ticker symbol for the selected company
    selected_symbol = name_to_symbol[selected_company]
    
    return selected_symbol, selected_company

# Sidebar for inputs
with st.sidebar:
    st.header("Stock Selection")

    # Inside your main app code or wherever you're handling stock selection:
    symbol, company = stock_selector()

    # Then use the symbol variable to fetch stock data or perform other operations
    # For example:
    st.write(f"Selected: {company} ({symbol})")

    # Use the symbol in your existing stock data fetching code
    # # stock_data = yf.download(symbol, ...)
    
    # If we have a temp ticker selected from Nifty 50, use it
    default_ticker = st.session_state.temp_ticker if st.session_state.temp_ticker else "RELIANCE.NS"
    ticker = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS)", default_ticker)
    
    
    # Provide a note for traders
    st.caption("All stock symbols must end with '.NS' for NSE (National Stock Exchange) listings")
    
    # Display common market indices as quick links
    st.subheader("Indian Market Indices")
    indices_cols = st.columns(2)
    
    # Main Indian Indices
    with indices_cols[0]:
        if st.button("NIFTY 50 (^NSEI)"):
            st.session_state.temp_ticker = "^NSEI"
            st.rerun()
        if st.button("NIFTY BANK (^NSEBANK)"):
            st.session_state.temp_ticker = "^NSEBANK"
            st.rerun()
    
    # Additional Indian Indices
    with indices_cols[1]:
        if st.button("SENSEX (^BSESN)"):
            st.session_state.temp_ticker = "^BSESN"
            st.rerun()
        if st.button("NIFTY IT (^CNXIT)"):
            st.session_state.temp_ticker = "^CNXIT"
            st.rerun()
    
    # More features for advanced users
    st.header("Date Range")
    today = datetime.now()
    one_year_ago = today - timedelta(days=365)
    
    # Date range selection
    date_cols = st.columns(2)
    with date_cols[0]:
        start_date = st.date_input("Start Date", one_year_ago)
    with date_cols[1]:
        end_date = st.date_input("End Date", today)
    
    # Quick date selection buttons
    period_cols = st.columns(3)
    with period_cols[0]:
        if st.button("1M"):
            st.session_state.start_date = today - timedelta(days=30)
            st.rerun()
    with period_cols[1]:
        if st.button("3M"):
            st.session_state.start_date = today - timedelta(days=90)
            st.rerun()
    with period_cols[2]:
        if st.button("1Y"):
            st.session_state.start_date = today - timedelta(days=365)
            st.rerun()
    
    # Validate date range
    if start_date >= end_date:
        st.error("Error: End date must be after start date.")
        st.stop()
    
    # Reset temporary ticker after fetch
    if st.button("Fetch Stock Data"):
        st.session_state.fetch_requested = True
        # Reset temp ticker after using it
        st.session_state.temp_ticker = None
    else:
        if 'fetch_requested' not in st.session_state:
            st.session_state.fetch_requested = False

# Main content
if st.session_state.fetch_requested:
    # Display loading message
    with st.spinner(f"Fetching data for {ticker}..."):
        # Get the data
        stock_data, stock_info = get_stock_data(ticker.upper(), start_date, end_date)
    
    if stock_data is not None and not stock_data.empty:
        # Success message
        st.success(f"Successfully retrieved data for {ticker.upper()}")
        
        # Add technical indicators to stock data
        stock_data_with_indicators = add_technical_indicators(stock_data)
        
        # Calculate support and resistance levels
        supports, resistances = calculate_support_resistance(stock_data)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Overview", 
            "Technical Analysis", 
            "Historical Data", 
            "Statistical",
            "Quantitative Analysis",
            "Advanced Analysis"
        ])
        
        with tab1:
            # Display stock info
            if stock_info:
                # Top metrics for at-a-glance view
                if 'longName' in stock_info:
                    st.subheader(f"{stock_info.get('longName', ticker.upper())}")
                
                # Price metrics with color coding
                price_metrics = st.container()
                with price_metrics:
                    metrics_cols = st.columns(4)
                    
                    # Set currency prefix to Indian Rupee symbol for all stocks
                    price_prefix = "₹"
                    
                    # Current price
                    if 'regularMarketPrice' in stock_info:
                        curr_price = stock_info.get('regularMarketPrice', 'N/A')
                        metrics_cols[0].metric("Current Price", 
                                               f"{price_prefix}{curr_price:,.2f}" if isinstance(curr_price, (int, float)) else "N/A")
                    
                    # Price change
                    if 'regularMarketChange' in stock_info and 'regularMarketChangePercent' in stock_info:
                        change = stock_info.get('regularMarketChange', 'N/A')
                        change_pct = stock_info.get('regularMarketChangePercent', 'N/A')
                        
                        if isinstance(change, (int, float)) and isinstance(change_pct, (int, float)):
                            change_str = f"{price_prefix}{change:+,.2f} ({change_pct:+,.2f}%)"
                            metrics_cols[1].metric("Today's Change", change_str, delta=change_pct)
                    
                    # 52-week high/low
                    if 'fiftyTwoWeekHigh' in stock_info and 'fiftyTwoWeekLow' in stock_info:
                        high_52wk = stock_info.get('fiftyTwoWeekHigh', 'N/A')
                        low_52wk = stock_info.get('fiftyTwoWeekLow', 'N/A')
                        
                        if isinstance(high_52wk, (int, float)) and isinstance(low_52wk, (int, float)):
                            # Calculate how close current price is to 52-week high
                            curr_price = stock_info.get('regularMarketPrice', 0)
                            if isinstance(curr_price, (int, float)) and curr_price > 0:
                                pct_of_high = (curr_price / high_52wk) * 100 - 100
                                high_low_str = f"{price_prefix}{low_52wk:,.2f} - {price_prefix}{high_52wk:,.2f}"
                                metrics_cols[2].metric("52-Week Range", high_low_str, f"{pct_of_high:,.2f}% from high")
                    
                    # Volume
                    if 'regularMarketVolume' in stock_info and 'averageVolume' in stock_info:
                        volume = stock_info.get('regularMarketVolume', 'N/A')
                        avg_volume = stock_info.get('averageVolume', 'N/A')
                        
                        if isinstance(volume, (int, float)) and isinstance(avg_volume, (int, float)) and avg_volume > 0:
                            vol_change = ((volume / avg_volume) - 1) * 100
                            metrics_cols[3].metric("Volume", f"{volume:,.0f}", f"{vol_change:+,.2f}% vs avg")
                
                # Company information and market data
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.subheader("Company Information")
                    if 'sector' in stock_info:
                        st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                    if 'industry' in stock_info:
                        st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
                    if 'fullTimeEmployees' in stock_info:
                        employees = stock_info.get('fullTimeEmployees', 'N/A')
                        if isinstance(employees, (int, float)):
                            st.write(f"**Employees:** {employees:,}")
                        else:
                            st.write(f"**Employees:** {employees}")
                    if 'website' in stock_info:
                        st.write(f"**Website:** [{stock_info.get('website', 'N/A')}]({stock_info.get('website', '#')})")
                    if 'country' in stock_info:
                        st.write(f"**Country:** {stock_info.get('country', 'N/A')}")
                
                with col2:
                    st.subheader("Key Financial Metrics")
                    if 'marketCap' in stock_info:
                        market_cap = stock_info.get('marketCap', 0)
                        if isinstance(market_cap, (int, float)):
                            if market_cap >= 1e12:  # trillion
                                st.write(f"**Market Cap:** {price_prefix}{market_cap/1e12:,.2f} Trillion")
                            else:
                                st.write(f"**Market Cap:** {price_prefix}{market_cap/1e9:,.2f} Billion")
                        else:
                            st.write(f"**Market Cap:** N/A")
                    
                    if 'trailingPE' in stock_info:
                        pe = stock_info.get('trailingPE', 'N/A')
                        if isinstance(pe, (int, float)):
                            st.write(f"**P/E Ratio:** {pe:,.2f}")
                        else:
                            st.write(f"**P/E Ratio:** N/A")
                    
                    if 'forwardPE' in stock_info:
                        forward_pe = stock_info.get('forwardPE', 'N/A')
                        if isinstance(forward_pe, (int, float)):
                            st.write(f"**Forward P/E:** {forward_pe:,.2f}")
                    
                    if 'priceToBook' in stock_info:
                        pb = stock_info.get('priceToBook', 'N/A')
                        if isinstance(pb, (int, float)):
                            st.write(f"**Price/Book:** {pb:,.2f}")
                    
                    if 'dividendYield' in stock_info and stock_info['dividendYield'] is not None:
                        dividend = stock_info.get('dividendYield', 0)
                        if isinstance(dividend, (int, float)):
                            st.write(f"**Dividend Yield:** {dividend * 100:.2f}%")
                        else:
                            st.write(f"**Dividend Yield:** N/A")
                            
                with col3:
                    st.subheader("Trading Info")
                    if 'previousClose' in stock_info:
                        prev_close = stock_info.get('previousClose', 'N/A')
                        if isinstance(prev_close, (int, float)):
                            st.write(f"**Prev Close:** {price_prefix}{prev_close:,.2f}")
                    
                    if 'open' in stock_info:
                        open_price = stock_info.get('open', 'N/A')
                        if isinstance(open_price, (int, float)):
                            st.write(f"**Open:** {price_prefix}{open_price:,.2f}")
                    
                    if 'dayHigh' in stock_info and 'dayLow' in stock_info:
                        day_high = stock_info.get('dayHigh', 'N/A')
                        day_low = stock_info.get('dayLow', 'N/A')
                        if isinstance(day_high, (int, float)) and isinstance(day_low, (int, float)):
                            st.write(f"**Day Range:** {price_prefix}{day_low:,.2f} - {price_prefix}{day_high:,.2f}")
                
                # Business summary if available
                if 'longBusinessSummary' in stock_info:
                    with st.expander("Business Summary"):
                        st.write(stock_info.get('longBusinessSummary', 'No business summary available.'))
            
            # Display stock price chart
            st.subheader("Stock Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Price'
            ))
            
            # Add volume bar chart at the bottom
            fig.add_trace(go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name='Volume',
                marker_color='rgba(0, 0, 255, 0.3)',
                opacity=0.3,
                yaxis='y2'
            ))
            
            # Set currency symbol to Indian Rupee
            currency_symbol = '₹'
            
            # Layout customization
            fig.update_layout(
                title=f'{ticker.upper()} Stock Price',
                xaxis_title='Date',
                yaxis_title=f'Price ({currency_symbol})',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            )
            
            # Make the chart responsive
            fig.update_layout(
                autosize=True,
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("Technical Analysis")
            
            # Create subtabs for different technical analysis types
            tech_tab1, tech_tab2, tech_tab3, tech_tab4 = st.tabs([
                "Price & Moving Averages", 
                "Oscillators & Momentum", 
                "Volatility Indicators",
                "Support & Resistance"
            ])
            
            with tech_tab1:
                st.write("### Price and Moving Averages")
                
                # Create a candlestick chart with moving averages
                fig_price = go.Figure()
                
                # Add candlestick
                fig_price.add_trace(go.Candlestick(
                    x=stock_data_with_indicators.index,
                    open=stock_data_with_indicators['Open'],
                    high=stock_data_with_indicators['High'],
                    low=stock_data_with_indicators['Low'],
                    close=stock_data_with_indicators['Close'],
                    name='Price'
                ))
                
                # Add moving averages with distinct colors
                fig_price.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index, 
                    y=stock_data_with_indicators['MA5'], 
                    line=dict(color='blue', width=1), 
                    name='MA5'
                ))
                fig_price.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index, 
                    y=stock_data_with_indicators['MA20'], 
                    line=dict(color='orange', width=1), 
                    name='MA20'
                ))
                fig_price.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index, 
                    y=stock_data_with_indicators['MA50'], 
                    line=dict(color='green', width=1.5), 
                    name='MA50'
                ))
                fig_price.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index, 
                    y=stock_data_with_indicators['MA200'], 
                    line=dict(color='red', width=2), 
                    name='MA200'
                ))
                
                # Add EMA
                fig_price.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index, 
                    y=stock_data_with_indicators['EMA9'], 
                    line=dict(color='purple', width=1, dash='dash'), 
                    name='EMA9'
                ))
                fig_price.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index, 
                    y=stock_data_with_indicators['EMA21'], 
                    line=dict(color='brown', width=1, dash='dash'), 
                    name='EMA21'
                ))
                
                # Update layout
                fig_price.update_layout(
                    title='Price with Moving Averages',
                    yaxis_title='Price (₹)',
                    xaxis_rangeslider_visible=False,
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Technical Analysis Insights
                st.subheader("Moving Average Analysis")
                
                # Calculate current values
                last_close = stock_data_with_indicators['Close'].iloc[-1]
                last_ma5 = stock_data_with_indicators['MA5'].iloc[-1]
                last_ma20 = stock_data_with_indicators['MA20'].iloc[-1]
                last_ma50 = stock_data_with_indicators['MA50'].iloc[-1]
                last_ma200 = stock_data_with_indicators['MA200'].iloc[-1]
                
                # Golden Cross / Death Cross detection (MA50 and MA200)
                ma50_series = stock_data_with_indicators['MA50'].dropna()
                ma200_series = stock_data_with_indicators['MA200'].dropna()
                
                # Ensure we have enough data
                if len(ma50_series) > 2 and len(ma200_series) > 2:
                    ma_cross_signal = ""
                    # Check for recent golden cross (MA50 crosses above MA200)
                    if ma50_series.iloc[-2] <= ma200_series.iloc[-2] and ma50_series.iloc[-1] > ma200_series.iloc[-1]:
                        ma_cross_signal = "🟢 **GOLDEN CROSS DETECTED**: MA50 has crossed above MA200, which is traditionally a bullish signal for long-term trend reversal."
                    # Check for recent death cross (MA50 crosses below MA200)
                    elif ma50_series.iloc[-2] >= ma200_series.iloc[-2] and ma50_series.iloc[-1] < ma200_series.iloc[-1]:
                        ma_cross_signal = "🔴 **DEATH CROSS DETECTED**: MA50 has crossed below MA200, which is traditionally a bearish signal for long-term trend reversal."
                    
                    if ma_cross_signal:
                        st.markdown(ma_cross_signal)
                
                # Create metrics for important MA relationships
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    ma_status = "Above" if last_close > last_ma50 else "Below"
                    pct_diff = ((last_close / last_ma50) - 1) * 100
                    st.metric("Price vs MA50", f"{ma_status} MA50", f"{pct_diff:.2f}%")
                
                with col2:
                    ma_trend = "Bullish" if last_ma20 > last_ma50 else "Bearish"
                    st.metric("MA20 vs MA50", ma_trend)
                
                with col3:
                    ma_long_trend = "Bullish" if last_ma50 > last_ma200 else "Bearish"
                    st.metric("MA50 vs MA200", ma_long_trend)
                
                # Overall MA trend analysis
                st.subheader("Moving Average Trend Analysis")
                
                # Determine short-term trend (using MA5 and MA20)
                short_trend = "Bullish" if last_ma5 > last_ma20 else "Bearish"
                
                # Determine medium-term trend (using MA20 and MA50)
                medium_trend = "Bullish" if last_ma20 > last_ma50 else "Bearish"
                
                # Determine long-term trend (using MA50 and MA200)
                long_trend = "Bullish" if last_ma50 > last_ma200 else "Bearish"
                
                # Display trend analysis
                st.write(f"**Short-term trend (MA5 vs MA20):** {short_trend}")
                st.write(f"**Medium-term trend (MA20 vs MA50):** {medium_trend}")
                st.write(f"**Long-term trend (MA50 vs MA200):** {long_trend}")
                
                # Overall trend assessment
                trend_count = sum([1 if trend == "Bullish" else 0 for trend in [short_trend, medium_trend, long_trend]])
                
                if trend_count == 3:
                    st.success("**Overall Trend: Strongly Bullish** - All timeframes showing bullish alignment")
                elif trend_count == 2:
                    st.info("**Overall Trend: Moderately Bullish** - Majority of timeframes showing bullish signals")
                elif trend_count == 1:
                    st.warning("**Overall Trend: Moderately Bearish** - Majority of timeframes showing bearish signals")
                else:
                    st.error("**Overall Trend: Strongly Bearish** - All timeframes showing bearish alignment")
            
            with tech_tab2:
                st.write("### Momentum Indicators")
                
                # Create a figure for RSI
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators['RSI'],
                    line=dict(color='purple', width=1.5),
                    name='RSI'
                ))
                
                # Add horizontal lines at 70 and 30 (overbought/oversold levels)
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Neutral")
                
                fig_rsi.update_layout(
                    title='Relative Strength Index (RSI)',
                    yaxis_title='RSI Value',
                    height=300,
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # RSI Analysis
                last_rsi = stock_data_with_indicators['RSI'].iloc[-1]
                rsi_signal = ""
                if last_rsi > 70:
                    rsi_signal = "🔴 **Overbought:** RSI above 70 suggests the stock may be overvalued."
                elif last_rsi < 30:
                    rsi_signal = "🟢 **Oversold:** RSI below 30 suggests the stock may be undervalued."
                else:
                    rsi_signal = "⚪ **Neutral:** RSI between 30-70 indicates neither overbought nor oversold conditions."
                
                st.write(f"**Current RSI Value:** {last_rsi:.2f}")
                st.write(rsi_signal)
                
                # Create a figure for MACD
                fig_macd = go.Figure()
                
                # Add MACD line
                fig_macd.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators['MACD'],
                    line=dict(color='blue', width=1.5),
                    name='MACD Line'
                ))
                
                # Add Signal line
                fig_macd.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators['MACD_Signal'],
                    line=dict(color='red', width=1.5),
                    name='Signal Line'
                ))
                
                # Add Histogram
                colors = ['green' if val >= 0 else 'red' for val in stock_data_with_indicators['MACD_Hist']]
                fig_macd.add_trace(go.Bar(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators['MACD_Hist'],
                    marker_color=colors,
                    name='Histogram'
                ))
                
                fig_macd.update_layout(
                    title='Moving Average Convergence Divergence (MACD)',
                    yaxis_title='MACD Value',
                    height=300
                )
                st.plotly_chart(fig_macd, use_container_width=True)
                
                # MACD Analysis
                last_macd = stock_data_with_indicators['MACD'].iloc[-1]
                last_signal = stock_data_with_indicators['MACD_Signal'].iloc[-1]
                last_hist = stock_data_with_indicators['MACD_Hist'].iloc[-1]
                
                macd_crossover = ""
                # Check for recent MACD crossover
                if stock_data_with_indicators['MACD'].iloc[-2] <= stock_data_with_indicators['MACD_Signal'].iloc[-2] and \
                   stock_data_with_indicators['MACD'].iloc[-1] > stock_data_with_indicators['MACD_Signal'].iloc[-1]:
                    macd_crossover = "🟢 **Bullish Crossover:** MACD line has crossed above the signal line, which is a potential buy signal."
                elif stock_data_with_indicators['MACD'].iloc[-2] >= stock_data_with_indicators['MACD_Signal'].iloc[-2] and \
                     stock_data_with_indicators['MACD'].iloc[-1] < stock_data_with_indicators['MACD_Signal'].iloc[-1]:
                    macd_crossover = "🔴 **Bearish Crossover:** MACD line has crossed below the signal line, which is a potential sell signal."
                
                st.write(f"**Current MACD:** {last_macd:.4f}, **Signal Line:** {last_signal:.4f}, **Histogram:** {last_hist:.4f}")
                if macd_crossover:
                    st.write(macd_crossover)
                
                # Stochastic Oscillator
                st.subheader("Stochastic Oscillator")
                
                fig_stoch = go.Figure()
                fig_stoch.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators['Stoch_K'],
                    line=dict(color='blue', width=1.5),
                    name='%K Line'
                ))
                fig_stoch.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators['Stoch_D'],
                    line=dict(color='red', width=1.5),
                    name='%D Line'
                ))
                
                # Add horizontal lines for overbought/oversold
                fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
                
                fig_stoch.update_layout(
                    title='Stochastic Oscillator',
                    yaxis_title='Value',
                    height=300,
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig_stoch, use_container_width=True)
                
                # Stochastic Analysis
                last_k = stock_data_with_indicators['Stoch_K'].iloc[-1]
                last_d = stock_data_with_indicators['Stoch_D'].iloc[-1]
                
                stoch_signal = ""
                if last_k > 80 and last_d > 80:
                    stoch_signal = "🔴 **Overbought:** Both %K and %D are above 80, suggesting potential reversal or pullback."
                elif last_k < 20 and last_d < 20:
                    stoch_signal = "🟢 **Oversold:** Both %K and %D are below 20, suggesting potential reversal or bounce."
                
                # Check for Stochastic crossover
                stoch_crossover = ""
                if stock_data_with_indicators['Stoch_K'].iloc[-2] <= stock_data_with_indicators['Stoch_D'].iloc[-2] and \
                   stock_data_with_indicators['Stoch_K'].iloc[-1] > stock_data_with_indicators['Stoch_D'].iloc[-1]:
                    stoch_crossover = "🟢 **Bullish Crossover:** %K has crossed above %D, which is a potential buy signal."
                elif stock_data_with_indicators['Stoch_K'].iloc[-2] >= stock_data_with_indicators['Stoch_D'].iloc[-2] and \
                     stock_data_with_indicators['Stoch_K'].iloc[-1] < stock_data_with_indicators['Stoch_D'].iloc[-1]:
                    stoch_crossover = "🔴 **Bearish Crossover:** %K has crossed below %D, which is a potential sell signal."
                
                st.write(f"**Current Stochastic Values:** %K: {last_k:.2f}, %D: {last_d:.2f}")
                if stoch_signal:
                    st.write(stoch_signal)
                if stoch_crossover:
                    st.write(stoch_crossover)
            
            with tech_tab3:
                st.write("### Volatility Indicators")
                
                # Bollinger Bands
                st.subheader("Bollinger Bands")
                
                fig_bb = go.Figure()
                
                # Add price line
                fig_bb.add_trace(go.Candlestick(
                    x=stock_data_with_indicators.index,
                    open=stock_data_with_indicators['Open'],
                    high=stock_data_with_indicators['High'],
                    low=stock_data_with_indicators['Low'],
                    close=stock_data_with_indicators['Close'],
                    name='Price'
                ))
                
                # Add Bollinger Bands
                fig_bb.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators['BB_Upper'],
                    line=dict(color='red', width=1),
                    name='Upper Band',
                    line_dash='dash'
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators['BB_Middle'],
                    line=dict(color='blue', width=1),
                    name='Middle Band (20-day MA)'
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators['BB_Lower'],
                    line=dict(color='green', width=1),
                    name='Lower Band',
                    line_dash='dash'
                ))
                
                fig_bb.update_layout(
                    title='Bollinger Bands',
                    yaxis_title='Price (₹)',
                    xaxis_rangeslider_visible=False,
                    height=400
                )
                st.plotly_chart(fig_bb, use_container_width=True)
                
                # Bollinger Bands Analysis
                last_close = stock_data_with_indicators['Close'].iloc[-1]
                last_upper = stock_data_with_indicators['BB_Upper'].iloc[-1]
                last_middle = stock_data_with_indicators['BB_Middle'].iloc[-1]
                last_lower = stock_data_with_indicators['BB_Lower'].iloc[-1]
                
                # Calculate % B (position within Bollinger Bands)
                percent_b = (last_close - last_lower) / (last_upper - last_lower) if (last_upper - last_lower) != 0 else 0.5
                
                # Bollinger Band squeeze (narrowing bands indicate low volatility, potential for big move)
                recent_band_width = (last_upper - last_lower) / last_middle
                historical_band_width = [(u - l) / m for u, l, m in zip(
                    stock_data_with_indicators['BB_Upper'][-20:],
                    stock_data_with_indicators['BB_Lower'][-20:],
                    stock_data_with_indicators['BB_Middle'][-20:]
                )]
                avg_historical_width = sum(historical_band_width) / len(historical_band_width)
                
                # Identify Bollinger Band signals
                bb_signal = ""
                if last_close > last_upper:
                    bb_signal = "🔴 **Overbought:** Price is above the upper Bollinger Band, suggesting excessive buying pressure."
                elif last_close < last_lower:
                    bb_signal = "🟢 **Oversold:** Price is below the lower Bollinger Band, suggesting excessive selling pressure."
                
                # Detect Bollinger Band squeeze (potential for volatility breakout)
                squeeze_signal = ""
                if recent_band_width < 0.8 * avg_historical_width:
                    squeeze_signal = "⚠️ **Bollinger Band Squeeze Detected:** The bands are narrowing, indicating low volatility. This often precedes a significant price movement."
                
                st.write(f"**Current Position within Bands (% B):** {percent_b:.2%}")
                st.write(f"**Current Band Width:** {recent_band_width:.4f} (Historical Average: {avg_historical_width:.4f})")
                
                if bb_signal:
                    st.write(bb_signal)
                if squeeze_signal:
                    st.write(squeeze_signal)
                
                # Average True Range (ATR) - Volatility indicator
                st.subheader("Average True Range (ATR)")
                
                fig_atr = go.Figure()
                fig_atr.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators['ATR'],
                    line=dict(color='orange', width=1.5),
                    name='ATR (14)'
                ))
                
                fig_atr.update_layout(
                    title='Average True Range (14-period)',
                    yaxis_title='ATR Value',
                    height=300
                )
                st.plotly_chart(fig_atr, use_container_width=True)
                
                # ATR Analysis
                last_atr = stock_data_with_indicators['ATR'].iloc[-1]
                last_price = stock_data_with_indicators['Close'].iloc[-1]
                atr_percentage = (last_atr / last_price) * 100
                
                # Compare current ATR with its historical average
                atr_20_avg = stock_data_with_indicators['ATR'][-20:].mean()
                atr_vs_avg = (last_atr / atr_20_avg - 1) * 100
                
                st.write(f"**Current ATR:** {last_atr:.2f} ({atr_percentage:.2f}% of current price)")
                
                if last_atr > atr_20_avg * 1.2:
                    st.write(f"🔺 **High Volatility:** ATR is {atr_vs_avg:.2f}% above its 20-day average, indicating increased volatility.")
                elif last_atr < atr_20_avg * 0.8:
                    st.write(f"🔻 **Low Volatility:** ATR is {abs(atr_vs_avg):.2f}% below its 20-day average, indicating reduced volatility.")
                else:
                    st.write(f"⚪ **Normal Volatility:** ATR is close to its 20-day average.")
                
                # Daily volatility (standard deviation of returns)
                st.subheader("Historical Volatility")
                
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators['Volatility'],
                    line=dict(color='purple', width=1.5),
                    name='20-day Volatility'
                ))
                
                fig_vol.update_layout(
                    title='20-day Historical Volatility (Standard Deviation of Returns)',
                    yaxis_title='Volatility (%)',
                    height=300
                )
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Volatility Analysis
                last_vol = stock_data_with_indicators['Volatility'].iloc[-1]
                vol_20_avg = stock_data_with_indicators['Volatility'][-20:].mean()
                vol_vs_avg = (last_vol / vol_20_avg - 1) * 100 if vol_20_avg != 0 else 0
                
                st.write(f"**Current Volatility:** {last_vol:.2f}%")
                
                if last_vol > vol_20_avg * 1.2 and vol_20_avg > 0:
                    st.write(f"🔺 **Elevated Volatility:** Current volatility is {vol_vs_avg:.2f}% above its recent average.")
                elif last_vol < vol_20_avg * 0.8 and vol_20_avg > 0:
                    st.write(f"🔻 **Subdued Volatility:** Current volatility is {abs(vol_vs_avg):.2f}% below its recent average.")
            
            with tech_tab4:
                st.write("### Support and Resistance Levels")
                
                # Display the identified support and resistance levels
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Support Levels")
                    
                    if supports:
                        # Sort supports from highest to lowest for better readability
                        sorted_supports = sorted(supports, key=lambda x: x[1], reverse=True)
                        
                        for i, (date, level) in enumerate(sorted_supports):
                            st.write(f"Support #{i+1}: ₹{level:.2f} ({date.strftime('%Y-%m-%d')})")
                    else:
                        st.write("No significant support levels identified in the selected date range.")
                
                with col2:
                    st.subheader("Resistance Levels")
                    
                    if resistances:
                        # Sort resistances from lowest to highest for better readability
                        sorted_resistances = sorted(resistances, key=lambda x: x[1])
                        
                        for i, (date, level) in enumerate(sorted_resistances):
                            st.write(f"Resistance #{i+1}: ₹{level:.2f} ({date.strftime('%Y-%m-%d')})")
                    else:
                        st.write("No significant resistance levels identified in the selected date range.")
                
                # Create a candlestick chart with support and resistance levels
                st.subheader("Price Chart with Support and Resistance")
                
                fig_sr = go.Figure()
                
                # Add candlestick
                fig_sr.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name='Price'
                ))
                
                # Add support levels
                for date, level in supports:
                    fig_sr.add_shape(
                        type="line",
                        x0=date,
                        y0=level,
                        x1=stock_data.index[-1],
                        y1=level,
                        line=dict(color="green", width=2, dash="dash"),
                    )
                
                # Add resistance levels
                for date, level in resistances:
                    fig_sr.add_shape(
                        type="line",
                        x0=date,
                        y0=level,
                        x1=stock_data.index[-1],
                        y1=level,
                        line=dict(color="red", width=2, dash="dash"),
                    )
                
                # Update layout
                fig_sr.update_layout(
                    title='Price with Support and Resistance Levels',
                    yaxis_title='Price (₹)',
                    xaxis_rangeslider_visible=False,
                    height=500,
                    showlegend=True
                )
                
                # Add annotations for support and resistance lines
                for date, level in supports:
                    fig_sr.add_annotation(
                        x=stock_data.index[-1],
                        y=level,
                        text=f"Support: ₹{level:.2f}",
                        showarrow=False,
                        yshift=10,
                        xshift=50,
                        bgcolor="rgba(0,255,0,0.3)"
                    )
                
                for date, level in resistances:
                    fig_sr.add_annotation(
                        x=stock_data.index[-1],
                        y=level,
                        text=f"Resistance: ₹{level:.2f}",
                        showarrow=False,
                        yshift=-10,
                        xshift=50,
                        bgcolor="rgba(255,0,0,0.3)"
                    )
                
                st.plotly_chart(fig_sr, use_container_width=True)
                
                # Nearest support and resistance analysis
                if supports and resistances:
                    last_close = stock_data['Close'].iloc[-1]
                    
                    # Find nearest support
                    supports_below = [s for _, s in supports if s < last_close]
                    nearest_support = max(supports_below) if supports_below else None
                    
                    # Find nearest resistance
                    resistances_above = [r for _, r in resistances if r > last_close]
                    nearest_resistance = min(resistances_above) if resistances_above else None
                    
                    st.subheader("Price Position Analysis")
                    
                    if nearest_support and nearest_resistance:
                        support_distance = ((last_close / nearest_support) - 1) * 100
                        resistance_distance = ((nearest_resistance / last_close) - 1) * 100
                        
                        # Risk-reward ratio
                        risk_reward = resistance_distance / support_distance if support_distance > 0 else float('inf')
                        
                        st.write(f"**Current Price:** ₹{last_close:.2f}")
                        st.write(f"**Nearest Support:** ₹{nearest_support:.2f} ({support_distance:.2f}% below current price)")
                        st.write(f"**Nearest Resistance:** ₹{nearest_resistance:.2f} ({resistance_distance:.2f}% above current price)")
                        st.write(f"**Risk-Reward Ratio:** {risk_reward:.2f}")
                        
                        # Trading insight based on risk-reward
                        if risk_reward > 2:
                            st.success(f"📈 **Favorable Risk-Reward** (1:{risk_reward:.2f}): Potential upside is {risk_reward:.2f}x the downside risk.")
                        elif risk_reward < 1:
                            st.error(f"📉 **Unfavorable Risk-Reward** (1:{risk_reward:.2f}): Potential upside is less than the downside risk.")
                        else:
                            st.info(f"⚖️ **Balanced Risk-Reward** (1:{risk_reward:.2f}): Potential upside is roughly equivalent to the downside risk.")
                    else:
                        if not nearest_support:
                            st.warning("No support levels detected below the current price in the selected date range.")
                        if not nearest_resistance:
                            st.warning("No resistance levels detected above the current price in the selected date range.")
            
        with tab3:
            st.subheader("Historical Data")
            
            # Display summary statistics
            st.write("### Summary Statistics")
            summary_stats = stock_data.describe()
            st.dataframe(summary_stats)
            
            # Display historical data table
            st.write("### Historical Data Table")
            st.dataframe(stock_data)
            
            # Provide download link
            st.markdown(get_csv_download_link(stock_data, f"{ticker}_data_{start_date}_to_{end_date}.csv", "Download Data as CSV"), unsafe_allow_html=True)
        
        with tab3:
            st.subheader("Key Financial Metrics")
            
            # Calculate additional metrics
            if not stock_data.empty:
                # Daily returns
                stock_data['Daily Return'] = stock_data['Close'].pct_change() * 100
                
                # Moving averages
                stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
                stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
                stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
                
                # Volatility (20-day rolling standard deviation)
                stock_data['Volatility'] = stock_data['Daily Return'].rolling(window=20).std()
                
                # Display the calculated metrics
                metrics_data = stock_data[['Close', 'Daily Return', 'MA5', 'MA20', 'MA50', 'Volatility']].dropna()
                st.dataframe(metrics_data)
                
                # Set currency symbol to Indian Rupee
                currency_symbol = '₹'
                
                # Plot moving averages
                st.subheader("Moving Averages")
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
                fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA5'], mode='lines', name='5-Day MA'))
                fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA20'], mode='lines', name='20-Day MA'))
                fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='50-Day MA'))
                
                fig_ma.update_layout(
                    title='Moving Averages',
                    xaxis_title='Date',
                    yaxis_title=f'Price ({currency_symbol})',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    autosize=True,
                    height=400
                )
                
                st.plotly_chart(fig_ma, use_container_width=True)
                
                # Plot volatility
                st.subheader("Price Volatility (20-day)")
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Volatility'], mode='lines', name='20-Day Volatility'))
                
                fig_vol.update_layout(
                    title='20-Day Rolling Volatility',
                    xaxis_title='Date',
                    yaxis_title='Volatility (%)',
                    autosize=True,
                    height=300
                )
                
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Provide metrics download
                st.markdown(get_csv_download_link(metrics_data, f"{ticker}_metrics_{start_date}_to_{end_date}.csv", "Download Metrics as CSV"), unsafe_allow_html=True)
        
        with tab4:
            st.subheader("Financial Metrics")
            
            # Create subtabs for different types of quantitative analyses
            quant_tab1, quant_tab2, quant_tab3 = st.tabs([
                "Statistical Analysis", 
                "Correlation Analysis", 
                "Risk Metrics"
            ])
            
            with quant_tab1:
                st.write("### Statistical Analysis")
                
                # Calculating returns and log returns
                if not stock_data.empty:
                    # Create a copy to avoid warnings
                    returns_df = stock_data.copy()
                    
                    # Calculate returns
                    returns_df['Daily Return'] = returns_df['Close'].pct_change() * 100
                    returns_df['Log Return'] = np.log(returns_df['Close'] / returns_df['Close'].shift(1)) * 100
                    
                    # Remove NaNs
                    returns_df = returns_df.dropna()
                    
                    # Display basic stats for returns
                    st.write("#### Returns Statistics")
                    
                    # Calculate key statistics
                    mean_return = returns_df['Daily Return'].mean()
                    median_return = returns_df['Daily Return'].median()
                    min_return = returns_df['Daily Return'].min()
                    max_return = returns_df['Daily Return'].max()
                    std_return = returns_df['Daily Return'].std()
                    skew_value = returns_df['Daily Return'].skew()
                    kurt_value = returns_df['Daily Return'].kurtosis()
                    
                    # Display statistics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Return (%)", f"{mean_return:.2f}")
                        st.metric("Std Deviation (%)", f"{std_return:.2f}")
                    
                    with col2:
                        st.metric("Min Return (%)", f"{min_return:.2f}")
                        st.metric("Max Return (%)", f"{max_return:.2f}")
                    
                    with col3:
                        st.metric("Skewness", f"{skew_value:.3f}")
                        st.metric("Kurtosis", f"{kurt_value:.3f}")
                    
                    # Return Distribution Visualization
                    st.write("#### Return Distribution")
                    
                    # Create histogram of returns
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=returns_df['Daily Return'],
                        name='Daily Returns',
                        opacity=0.7,
                        marker_color='blue',
                        nbinsx=30
                    ))
                    
                    # Add a normal distribution curve for comparison
                    x_range = np.linspace(min_return, max_return, 100)
                    norm_pdf = (1/(std_return * np.sqrt(2 * np.pi))) * np.exp(-(x_range - mean_return)**2 / (2 * std_return**2))
                    # Scale the normal curve to match histogram height
                    scale_factor = len(returns_df['Daily Return']) * (max_return - min_return) / 30
                    
                    fig_hist.add_trace(go.Scatter(
                        x=x_range,
                        y=norm_pdf * scale_factor,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_hist.update_layout(
                        title='Distribution of Daily Returns',
                        xaxis_title='Daily Return (%)',
                        yaxis_title='Frequency',
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Normality Test Analysis
                    st.write("#### Normality Analysis")
                    st.write("""
                    The skewness and kurtosis values provide insights into the shape of the return distribution:
                    
                    - **Skewness** measures the asymmetry of the distribution. A value of zero indicates symmetry.
                      - Negative skew (< 0): Left tail is longer (more extreme negative returns)
                      - Positive skew (> 0): Right tail is longer (more extreme positive returns)
                    
                    - **Kurtosis** measures the "tailedness" of the distribution. Higher values indicate more extreme outliers.
                      - Normal distribution has kurtosis of 3.0 (excess kurtosis of 0)
                      - Higher kurtosis means more frequent extreme events than predicted by normal distribution
                    """)
                    
                    # Interpret the specific values
                    if abs(skew_value) < 0.5:
                        skew_interp = "The returns show relatively symmetric distribution."
                    elif skew_value < -0.5:
                        skew_interp = "The returns are negatively skewed, with more extreme negative returns than positive ones."
                    else:
                        skew_interp = "The returns are positively skewed, with more extreme positive returns than negative ones."
                    
                    if abs(kurt_value) < 0.5:
                        kurt_interp = "The return distribution has tails similar to a normal distribution."
                    elif kurt_value > 0.5:
                        kurt_interp = "The return distribution has heavy tails (leptokurtic), suggesting more frequent extreme returns."
                    else:
                        kurt_interp = "The return distribution has light tails (platykurtic), suggesting less frequent extreme returns."
                    
                    st.write(f"**Interpretation:** {skew_interp} {kurt_interp}")
                    
                    # Autocorrelation Analysis
                    st.write("#### Return Autocorrelation")
                    st.write("Autocorrelation measures the correlation between a time series and a lagged version of itself.")
                    
                    # Calculate autocorrelation for daily returns
                    autocorr_values = []
                    lags = range(1, 11)  # Calculate for lags 1 to 10
                    
                    for lag in lags:
                        autocorr = returns_df['Daily Return'].autocorr(lag=lag)
                        autocorr_values.append(autocorr)
                    
                    # Create autocorrelation plot
                    fig_autocorr = go.Figure()
                    fig_autocorr.add_trace(go.Bar(
                        x=list(lags),
                        y=autocorr_values,
                        marker_color=['blue' if ac > 0 else 'red' for ac in autocorr_values]
                    ))
                    
                    # Add significant level lines
                    n = len(returns_df)
                    # 95% confidence interval for autocorrelation (approximate)
                    conf_level = 1.96 / np.sqrt(n)
                    
                    fig_autocorr.add_shape(
                        type='line',
                        x0=0.5, x1=10.5,
                        y0=conf_level, y1=conf_level,
                        line=dict(color='grey', width=2, dash='dash')
                    )
                    
                    fig_autocorr.add_shape(
                        type='line',
                        x0=0.5, x1=10.5,
                        y0=-conf_level, y1=-conf_level,
                        line=dict(color='grey', width=2, dash='dash')
                    )
                    
                    fig_autocorr.update_layout(
                        title='Autocorrelation of Daily Returns',
                        xaxis_title='Lag (Days)',
                        yaxis_title='Autocorrelation',
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_autocorr, use_container_width=True)
                    
                    # Interpretation of autocorrelation
                    significant_lags = [lag+1 for lag, ac in enumerate(autocorr_values) if abs(ac) > conf_level]
                    
                    if significant_lags:
                        st.write(f"**Significant autocorrelation detected at lags: {significant_lags}**")
                        st.write("""
                        The presence of significant autocorrelation suggests that past returns have predictive power for future returns, 
                        which may indicate market inefficiency or momentum/reversal patterns.
                        """)
                    else:
                        st.write("""
                        **No significant autocorrelation detected.**
                        
                        The absence of significant autocorrelation is consistent with the Efficient Market Hypothesis, 
                        which suggests that past returns cannot be used to predict future returns.
                        """)
            
            with quant_tab2:
                st.write("### Correlation Analysis")
                
                # Create a dropdown to select benchmark indices for correlation
                benchmark_options = {
                    "^NSEI": "NIFTY 50",
                    "^BSESN": "SENSEX",
                    "^NSEBANK": "NIFTY BANK",
                    "^CNXIT": "NIFTY IT",
                    "RELIANCE.NS": "Reliance Industries",
                    "TCS.NS": "Tata Consultancy Services",
                    "HDFCBANK.NS": "HDFC Bank"
                }
                
                # Add sector specific indices if available
                if stock_info and 'sector' in stock_info:
                    sector = stock_info.get('sector', '').strip()
                    if sector == "Financial Services":
                        benchmark_options["^NSEBANK"] = "NIFTY BANK"
                    elif sector == "Information Technology":
                        benchmark_options["^CNXIT"] = "NIFTY IT"
                    elif sector == "Energy":
                        benchmark_options["^CNXENERGY"] = "NIFTY ENERGY"
                    elif sector == "Healthcare":
                        benchmark_options["^CNXPHARMA"] = "NIFTY PHARMA"
                    elif sector == "Consumer Goods":
                        benchmark_options["^CNXFMCG"] = "NIFTY FMCG"
                
                # Select benchmarks to compare with
                st.write("#### Select Benchmarks for Correlation")
                selected_benchmarks = []
                
                col1, col2 = st.columns(2)
                with col1:
                    # Always include Nifty 50 as default benchmark
                    if st.checkbox("NIFTY 50 (^NSEI)", value=True):
                        selected_benchmarks.append("^NSEI")
                    
                    if st.checkbox("SENSEX (^BSESN)"):
                        selected_benchmarks.append("^BSESN")
                    
                    if st.checkbox("NIFTY BANK (^NSEBANK)"):
                        selected_benchmarks.append("^NSEBANK")
                    
                with col2:
                    if st.checkbox("NIFTY IT (^CNXIT)"):
                        selected_benchmarks.append("^CNXIT")
                    
                    if st.checkbox("Reliance Industries (RELIANCE.NS)"):
                        selected_benchmarks.append("RELIANCE.NS")
                    
                    if st.checkbox("HDFC Bank (HDFCBANK.NS)"):
                        selected_benchmarks.append("HDFCBANK.NS")
                
                if selected_benchmarks:
                    with st.spinner("Calculating correlations..."):
                        # Perform correlation analysis
                        correlation_matrix, beta = get_correlation_analysis(ticker, selected_benchmarks, start_date, end_date)
                        
                        if correlation_matrix is not None and not correlation_matrix.empty:
                            # Display correlation matrix as a heatmap
                            st.write("#### Correlation Matrix")
                            
                            # Create a more visually appealing correlation heatmap
                            fig_corr = px.imshow(
                                correlation_matrix,
                                text_auto=True,
                                color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1,
                                aspect="auto"
                            )
                            
                            fig_corr.update_layout(
                                title="Correlation Matrix of Returns",
                                height=400
                            )
                            
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            # Display correlation values in a cleaner table format
                            st.write("#### Correlation with Selected Benchmarks")
                            
                            # Extract correlations with the stock
                            if ticker in correlation_matrix.columns:
                                correlations = correlation_matrix[ticker].drop(ticker)
                                
                                # Create a DataFrame for better presentation
                                corr_df = pd.DataFrame({
                                    'Benchmark': [benchmark_options.get(idx, idx) for idx in correlations.index],
                                    'Correlation': correlations.values
                                })
                                
                                # Format the correlation values
                                corr_df['Correlation'] = corr_df['Correlation'].apply(lambda x: f"{x:.4f}")
                                
                                # Display the correlations
                                st.dataframe(corr_df)
                                
                                # Interpretation of correlations
                                max_corr_idx = correlations.abs().idxmax()
                                max_corr_val = correlations[max_corr_idx]
                                max_corr_name = benchmark_options.get(max_corr_idx, max_corr_idx)
                                
                                if abs(max_corr_val) > 0.7:
                                    st.write(f"**Strong {'positive' if max_corr_val > 0 else 'negative'} correlation with {max_corr_name}** ({max_corr_val:.4f})")
                                elif abs(max_corr_val) > 0.3:
                                    st.write(f"**Moderate {'positive' if max_corr_val > 0 else 'negative'} correlation with {max_corr_name}** ({max_corr_val:.4f})")
                                else:
                                    st.write(f"**Weak correlation with all selected benchmarks** (highest: {max_corr_val:.4f} with {max_corr_name})")
                            
                            # Display Beta
                            if beta is not None and selected_benchmarks:
                                st.write("#### Beta (β)")
                                st.metric(
                                    f"Beta relative to {benchmark_options.get(selected_benchmarks[0], selected_benchmarks[0])}",
                                    f"{beta:.4f}"
                                )
                                
                                # Interpret Beta
                                if beta > 1.5:
                                    st.write(f"**High Beta (β = {beta:.4f})**: The stock is much more volatile than the benchmark. For every 1% change in the benchmark, the stock tends to move approximately {beta:.2f}%.")
                                elif beta > 1.1:
                                    st.write(f"**Above Average Beta (β = {beta:.4f})**: The stock is more volatile than the benchmark. For every 1% change in the benchmark, the stock tends to move approximately {beta:.2f}%.")
                                elif beta > 0.9:
                                    st.write(f"**Average Beta (β = {beta:.4f})**: The stock moves similarly to the benchmark. For every 1% change in the benchmark, the stock tends to move approximately {beta:.2f}%.")
                                elif beta > 0:
                                    st.write(f"**Below Average Beta (β = {beta:.4f})**: The stock is less volatile than the benchmark. For every 1% change in the benchmark, the stock tends to move approximately {beta:.2f}%.")
                                elif beta < 0:
                                    st.write(f"**Negative Beta (β = {beta:.4f})**: The stock tends to move in the opposite direction of the benchmark. For every 1% change in the benchmark, the stock tends to move approximately {abs(beta):.2f}% in the opposite direction.")
                        else:
                            st.warning("Could not calculate correlations. Please try a different date range or benchmarks.")
                else:
                    st.warning("Please select at least one benchmark for correlation analysis.")
            
            with quant_tab3:
                st.write("### Risk Metrics")
                
                if not stock_data.empty:
                    # Calculate daily returns for risk analysis
                    risk_df = stock_data.copy()
                    risk_df['Daily Return'] = risk_df['Close'].pct_change()
                    risk_df = risk_df.dropna()
                    
                    # Calculate key risk metrics
                    daily_returns = risk_df['Daily Return']
                    
                    # Annual return calculations
                    annual_return = daily_returns.mean() * 252 * 100  # 252 trading days in a year
                    
                    # Volatility (annualized)
                    volatility = daily_returns.std() * np.sqrt(252) * 100
                    
                    # Downside risk metrics
                    negative_returns = daily_returns[daily_returns < 0]
                    
                    # Maximum Drawdown
                    cumulative = (1 + risk_df['Daily Return']).cumprod()
                    max_drawdown = ((cumulative / cumulative.cummax()) - 1).min() * 100
                    
                    # Sharpe Ratio (Assuming risk-free rate of 4% for Indian market)
                    risk_free_rate = 0.04  # 4% annual risk-free rate (adjust as needed)
                    daily_rf = ((1 + risk_free_rate) ** (1/252)) - 1
                    sharpe_ratio = (daily_returns.mean() - daily_rf) / daily_returns.std() * np.sqrt(252)
                    
                    # Sortino Ratio (using downside deviation)
                    target_return = daily_rf  # Usually risk-free rate
                    downside_returns = daily_returns[daily_returns < target_return]
                    downside_deviation = downside_returns.std() * np.sqrt(252)
                    sortino_ratio = (daily_returns.mean() - daily_rf) / downside_deviation if len(downside_returns) > 0 and downside_deviation > 0 else 0
                    
                    # Value at Risk (VaR) - 95% confidence
                    var_95 = np.percentile(daily_returns, 5) * 100
                    
                    # Conditional VaR / Expected Shortfall (CVaR/ES) - 95% confidence
                    cvar_95 = negative_returns[negative_returns <= np.percentile(daily_returns, 5)].mean() * 100
                    
                    # Display risk metrics
                    st.write("#### Key Risk Metrics")
                    
                    # Use columns for better organization
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Annual Return (%)", f"{annual_return:.2f}")
                        st.metric("Annual Volatility (%)", f"{volatility:.2f}")
                        st.metric("Maximum Drawdown (%)", f"{max_drawdown:.2f}")
                    
                    with col2:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.4f}")
                        st.metric("Sortino Ratio", f"{sortino_ratio:.4f}")
                    
                    with col3:
                        st.metric("Value at Risk (95%)", f"{var_95:.2f}%")
                        st.metric("Conditional VaR (95%)", f"{cvar_95:.2f}%" if not np.isnan(cvar_95) else "N/A")
                    
                    # Interpretations
                    st.write("#### Risk Metric Interpretations")
                    
                    # Sharpe Ratio interpretation
                    st.write("**Sharpe Ratio:** Measures risk-adjusted return (excess return per unit of risk)")
                    if sharpe_ratio > 1.0:
                        st.success(f"Sharpe Ratio of {sharpe_ratio:.2f} indicates good risk-adjusted returns relative to the risk-free rate.")
                    elif sharpe_ratio > 0:
                        st.info(f"Sharpe Ratio of {sharpe_ratio:.2f} indicates returns exceeding the risk-free rate, but consider if the additional risk is justified.")
                    else:
                        st.error(f"Sharpe Ratio of {sharpe_ratio:.2f} indicates the investment has underperformed compared to the risk-free rate.")
                    
                    # Sortino Ratio interpretation
                    st.write("**Sortino Ratio:** Similar to Sharpe but only considers downside risk (harmful volatility)")
                    if sortino_ratio > 1.0:
                        st.success(f"Sortino Ratio of {sortino_ratio:.2f} indicates good returns relative to the downside risk.")
                    elif sortino_ratio > 0:
                        st.info(f"Sortino Ratio of {sortino_ratio:.2f} indicates positive returns relative to downside risk, but may not be optimal.")
                    else:
                        st.error(f"Sortino Ratio of {sortino_ratio:.2f} indicates poor returns relative to downside risk.")
                    
                    # VaR interpretation
                    st.write("**Value at Risk (VaR):** Represents the maximum expected daily loss with 95% confidence")
                    st.info(f"With 95% confidence, the worst daily loss should not exceed {abs(var_95):.2f}%.")
                    
                    # Maximum Drawdown interpretation
                    st.write("**Maximum Drawdown:** Represents the maximum observed loss from a peak to a trough")
                    if max_drawdown > -10:
                        st.success(f"Maximum drawdown of {abs(max_drawdown):.2f}% indicates relatively low downside risk.")
                    elif max_drawdown > -20:
                        st.warning(f"Maximum drawdown of {abs(max_drawdown):.2f}% indicates moderate downside risk.")
                    else:
                        st.error(f"Maximum drawdown of {abs(max_drawdown):.2f}% indicates significant downside risk.")
                    
                    # Monte Carlo Simulation for potential future paths
                    st.write("#### Monte Carlo Simulation")
                    st.write("This simulation projects potential future price paths based on historical volatility and returns.")
                    
                    # Number of simulations
                    num_simulations = 100
                    num_days = 252  # Simulate one year ahead
                    
                    # Get parameters for simulation from historical data
                    mu = daily_returns.mean()
                    sigma = daily_returns.std()
                    
                    # Last closing price as starting point
                    last_price = stock_data['Close'].iloc[-1]
                    
                    # Run Monte Carlo simulation
                    simulation_df = pd.DataFrame()
                    
                    # Create a progress bar for simulation
                    with st.spinner("Running Monte Carlo simulation..."):
                        # Generate random returns
                        for i in range(num_simulations):
                            # Create a list for the price series
                            prices = [last_price]
                            
                            # Generate future prices
                            for j in range(num_days):
                                # Generate random return from normal distribution
                                daily_return = np.random.normal(mu, sigma)
                                
                                # Calculate next price
                                next_price = prices[-1] * (1 + daily_return)
                                prices.append(next_price)
                            
                            # Add the price series to the dataframe
                            simulation_df[f'Sim_{i}'] = prices
                    
                    # Create a plot of the simulations
                    fig_mc = go.Figure()
                    
                    # Plot a subset of simulations for better visibility
                    for i in range(0, num_simulations, 5):  # Plot every 5th simulation
                        fig_mc.add_trace(go.Scatter(
                            y=simulation_df[f'Sim_{i}'],
                            mode='lines',
                            line=dict(width=0.5),
                            showlegend=False
                        ))
                    
                    # Calculate percentiles for confidence intervals
                    perc_5 = simulation_df.quantile(0.05, axis=1)
                    perc_95 = simulation_df.quantile(0.95, axis=1)
                    median = simulation_df.quantile(0.5, axis=1)
                    
                    # Add percentile lines
                    fig_mc.add_trace(go.Scatter(
                        y=perc_5,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='5th Percentile'
                    ))
                    
                    fig_mc.add_trace(go.Scatter(
                        y=perc_95,
                        mode='lines',
                        line=dict(color='green', width=2),
                        name='95th Percentile'
                    ))
                    
                    fig_mc.add_trace(go.Scatter(
                        y=median,
                        mode='lines',
                        line=dict(color='blue', width=2, dash='dash'),
                        name='Median'
                    ))
                    
                    # Add the starting price as a horizontal line
                    fig_mc.add_shape(
                        type="line",
                        x0=0,
                        y0=last_price,
                        x1=num_days,
                        y1=last_price,
                        line=dict(color="black", width=1.5, dash="dot"),
                    )
                    
                    fig_mc.update_layout(
                        title='Monte Carlo Simulation - Projected Price Paths (1 Year)',
                        xaxis_title='Trading Days',
                        yaxis_title='Price (₹)',
                        height=500,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Display simulation statistics
                    st.write("#### Price Projection Statistics (1 Year)")
                    
                    # Calculate expected range
                    final_prices = simulation_df.iloc[-1]
                    final_mean = final_prices.mean()
                    final_std = final_prices.std()
                    final_min = final_prices.min()
                    final_max = final_prices.max()
                    
                    # Calculate the potential return statistics
                    potential_returns = (final_prices / last_price - 1) * 100
                    
                    # Display statistics in columns
                    stat_col1, stat_col2 = st.columns(2)
                    
                    with stat_col1:
                        st.metric("Current Price", f"₹{last_price:.2f}")
                        st.metric("Expected Mean Price", f"₹{final_mean:.2f}")
                        st.metric("Expected Return", f"{(final_mean/last_price - 1) * 100:.2f}%")
                    
                    with stat_col2:
                        st.metric("95% High Estimate", f"₹{perc_95.iloc[-1]:.2f}")
                        st.metric("95% Low Estimate", f"₹{perc_5.iloc[-1]:.2f}")
                        st.metric("Price Range", f"₹{final_min:.2f} - ₹{final_max:.2f}")
                    
                    # Probability of profit
                    prob_profit = (potential_returns > 0).mean() * 100
                    
                    st.write(f"**Probability of Positive Return:** {prob_profit:.1f}%")
                    
                    st.write("**Note:** This simulation is based on historical volatility and assumes returns follow a normal distribution. Actual results may vary due to market conditions, company-specific events, and other factors not captured in the model.")

        with tab5:
            st.subheader("Quantitative Analysis")

            # Create two columns for different analysis types
            quant_col1, quant_col2 = st.columns(2)

            with quant_col1:
                st.write("### Seasonality Analysis")
                
                # Extract month and day of week information
                stock_data['Month'] = stock_data.index.month
                stock_data['Day_of_Week'] = stock_data.index.dayofweek
                
                # Monthly Returns Analysis
                monthly_returns = stock_data.groupby('Month')['Close'].pct_change().groupby(stock_data['Month']).mean() * 100
                
                # Create a dataframe for monthly seasonality
                monthly_df = pd.DataFrame(monthly_returns).reset_index()
                monthly_df['Month'] = monthly_df['Month'].apply(lambda x: calendar.month_name[x])
                monthly_df.columns = ['Month', 'Average Return (%)']
                monthly_df = monthly_df.sort_values('Average Return (%)', ascending=False)
                
                st.write("#### Monthly Seasonality")
                fig = px.bar(monthly_df, x='Month', y='Average Return (%)', 
                             title=f"Average Monthly Returns for {ticker}",
                             color='Average Return (%)',
                             color_continuous_scale=['red', 'green'],
                             labels={'Average Return (%)': 'Avg Return (%)'})
                
                fig.update_layout(xaxis_title="Month", yaxis_title="Average Return (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Best and worst months
                st.write(f"**Best Month:** {monthly_df.iloc[0]['Month']} ({monthly_df.iloc[0]['Average Return (%)']:.2f}%)")
                st.write(f"**Worst Month:** {monthly_df.iloc[-1]['Month']} ({monthly_df.iloc[-1]['Average Return (%)']:.2f}%)")
                
            with quant_col2:
                st.write("### Return Distribution Analysis")

                # Calculate daily and monthly returns for distribution analysis
                daily_returns = stock_data['Close'].pct_change().dropna() * 100
                
                # Fit a normal distribution
                mu, std = norm.fit(daily_returns)
                
                # Create histogram with normal distribution overlay
                fig = px.histogram(daily_returns, 
                                  nbins=50, 
                                  title=f"Return Distribution Analysis for {ticker}",
                                  labels={'value': 'Daily Return (%)', 'count': 'Frequency'},
                                  opacity=0.7,
                                  color_discrete_sequence=['lightblue'])
                
                # Add normal distribution curve
                x = np.linspace(daily_returns.min(), daily_returns.max(), 100)
                y = norm.pdf(x, mu, std) * len(daily_returns) * (daily_returns.max() - daily_returns.min()) / 50
                
                fig.add_scatter(x=x, y=y, mode='lines', name='Normal Distribution', line=dict(color='red'))
                fig.update_layout(showlegend=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Return statistics
                st.write("#### Return Statistics")
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    st.metric("Mean Return (%)", f"{mu:.2f}%")
                    st.metric("Standard Deviation (%)", f"{std:.2f}%")
                    
                with stat_col2:
                    # Import skew and kurtosis functions directly from scipy.stats to avoid naming conflicts
                    from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
                    
                    # Calculate skewness and kurtosis
                    skewness = scipy_skew(daily_returns)
                    kurt = scipy_kurtosis(daily_returns)
                    
                    st.metric("Skewness", f"{skewness:.2f}")
                    st.metric("Kurtosis", f"{kurt:.2f}")
                
                st.write(f"**Interpretation:** Skewness of {skewness:.2f} indicates {'positive skew (more extreme positive returns)' if skewness > 0 else 'negative skew (more extreme negative returns)'}. Kurtosis of {kurt:.2f} suggests {'fatter tails than normal distribution (more outliers)' if kurt > 0 else 'thinner tails than normal distribution (fewer outliers)'}.")
                
            # Hurst Exponent Analysis (Bottom of the page)
            st.write("### Market Efficiency Analysis - Hurst Exponent")
            
            def hurst_exponent(time_series, max_lag=20):
                """Calculates the Hurst Exponent for the time series"""
                lags = range(2, max_lag)
                tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0]  # The Hurst exponent is the slope
            
            # Calculate the Hurst exponent
            prices = stock_data['Close'].values
            h_exponent = hurst_exponent(prices)
            
            st.write(f"**Hurst Exponent:** {h_exponent:.4f}")
            
            # Interpretation
            if h_exponent < 0.45:
                st.write("**Interpretation:** The price series exhibits mean-reverting behavior (H < 0.5). This suggests that price movements tend to revert to a mean value, potentially offering mean-reversion trading opportunities.")
            elif h_exponent > 0.55:
                st.write("**Interpretation:** The price series exhibits trend-following behavior (H > 0.5). This suggests that trends tend to persist, potentially offering trend-following trading opportunities.")
            else:
                st.write("**Interpretation:** The price series approximates a random walk (H ≈ 0.5), suggesting an efficient market where future price movements are difficult to predict based on past prices alone.")

        with tab6:
            st.subheader("Advanced Analysis")
            
            # Create tabs for different advanced analysis types
            adv_tab1, adv_tab2, adv_tab3 = st.tabs([
                "Event Analysis", 
                "Pair Trading Analysis", 
                "Volatility Analysis"
            ])
            
            with adv_tab1:
                st.write("### Event-Driven Analysis")
                
                # Historical Events Impact
                st.write("#### Impact of Major Market Events")
                
                # Define some major market events
                events = {
                    "COVID-19 Crash": ("2020-02-20", "2020-03-23"),
                    "2022 Market Correction": ("2022-01-01", "2022-06-20"),
                    "2023 Banking Crisis": ("2023-03-01", "2023-03-15"),
                    "2022 Ukraine Conflict": ("2022-02-24", "2022-03-15")
                }
                
                # Select an event to analyze
                selected_event = st.selectbox("Select Market Event to Analyze", list(events.keys()))
                
                # Get event dates
                event_start, event_end = events[selected_event]
                
                # Try to get data for the event period and one month before for comparison
                try:
                    # One month before event
                    pre_event_start = (pd.to_datetime(event_start) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
                    
                    # Get data
                    event_data, _ = get_stock_data(ticker, pre_event_start, event_end)
                    
                    if event_data is not None and not event_data.empty:
                        # Normalize to 100 at the event start date
                        event_start_idx = event_data.index[event_data.index >= event_start][0]
                        event_data['Normalized'] = event_data['Close'] / event_data.loc[event_start_idx, 'Close'] * 100
                        
                        # Get NIFTY 50 data for comparison
                        nifty_data, _ = get_stock_data("^NSEI", pre_event_start, event_end)
                        if nifty_data is not None and not nifty_data.empty:
                            nifty_data['Normalized'] = nifty_data['Close'] / nifty_data.loc[event_start_idx, 'Close'] * 100
                            
                            # Plot performance
                            fig = px.line(title=f"Performance During {selected_event}")
                            fig.add_scatter(x=event_data.index, y=event_data['Normalized'], name=ticker)
                            fig.add_scatter(x=nifty_data.index, y=nifty_data['Normalized'], name="NIFTY 50")
                            
                            # Add vertical lines for event start and end
                            fig.add_vline(x=event_start, line_dash="dash", line_color="red", annotation_text="Event Start")
                            fig.add_vline(x=event_end, line_dash="dash", line_color="green", annotation_text="Event End")
                            
                            fig.update_layout(xaxis_title="Date", yaxis_title="Normalized Price (100 = Event Start)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate returns during the event
                            event_return = (event_data.loc[event_data.index <= event_end, 'Normalized'].iloc[-1] / 100 - 1) * 100
                            nifty_return = (nifty_data.loc[nifty_data.index <= event_end, 'Normalized'].iloc[-1] / 100 - 1) * 100
                            
                            # Display metrics
                            event_col1, event_col2 = st.columns(2)
                            with event_col1:
                                st.metric(f"{ticker} Return", f"{event_return:.2f}%")
                            with event_col2:
                                st.metric("NIFTY 50 Return", f"{nifty_return:.2f}%")
                            
                            st.write(f"**Relative Performance:** {ticker} {'outperformed' if event_return > nifty_return else 'underperformed'} NIFTY 50 by {abs(event_return - nifty_return):.2f}% during this event.")
                            
                            # Volatility comparison
                            ticker_vol = event_data.loc[event_data.index >= event_start, 'Close'].pct_change().std() * np.sqrt(252) * 100
                            nifty_vol = nifty_data.loc[nifty_data.index >= event_start, 'Close'].pct_change().std() * np.sqrt(252) * 100
                            
                            st.write(f"**Volatility During Event:** {ticker}: {ticker_vol:.2f}%, NIFTY 50: {nifty_vol:.2f}%")
                            st.write(f"**Volatility Ratio:** {ticker_vol/nifty_vol:.2f}x NIFTY 50 volatility")
                        else:
                            st.error("Could not fetch NIFTY 50 data for comparison.")
                    else:
                        st.error(f"No data available for {ticker} during this event period.")
                except Exception as e:
                    st.error(f"Error analyzing event data: {e}")
                    
            with adv_tab2:
                st.write("### Pair Trading Analysis")
                
                # Allow user to select a pair stock
                st.write("#### Select a stock to pair with " + ticker)
                
                # Get Nifty 50 stocks
                nifty_stocks = get_nifty50_stocks()
                
                # Create a list of stocks excluding the current ticker
                pair_stock_options = [stock for stock in nifty_stocks.keys() if stock != ticker]
                    
                # Select a pair stock
                pair_stock = st.selectbox("Select pair stock", pair_stock_options)
                
                # Fetch data for both stocks
                try:
                    # Get data for pair stock
                    pair_data, _ = get_stock_data(pair_stock, start_date, end_date)
                    
                    if pair_data is not None and not pair_data.empty and not stock_data.empty:
                        # Merge dataframes
                        pair_df = pd.DataFrame({
                            f"{ticker}": stock_data['Close'],
                            f"{pair_stock}": pair_data['Close']
                        })
                        
                        # Calculate correlation
                        correlation = pair_df.corr().iloc[0, 1]
                        
                        # Display correlation
                        st.metric("Correlation", f"{correlation:.4f}")
                        
                        # Normalize prices for visualization
                        normalized_df = pd.DataFrame({
                            f"{ticker}": pair_df[ticker] / pair_df[ticker].iloc[0] * 100,
                            f"{pair_stock}": pair_df[pair_stock] / pair_df[pair_stock].iloc[0] * 100
                        })
                        
                        # Plot normalized prices
                        fig = px.line(title=f"Normalized Price Comparison: {ticker} vs {pair_stock}")
                        fig.add_scatter(x=normalized_df.index, y=normalized_df[ticker], name=ticker)
                        fig.add_scatter(x=normalized_df.index, y=normalized_df[pair_stock], name=pair_stock)
                        fig.update_layout(xaxis_title="Date", yaxis_title="Normalized Price (100 = Start)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate price ratio
                        pair_df['Ratio'] = pair_df[ticker] / pair_df[pair_stock]
                        
                        # Z-score of ratio
                        pair_df['Ratio_Z'] = (pair_df['Ratio'] - pair_df['Ratio'].mean()) / pair_df['Ratio'].std()
                        
                        # Plot ratio and z-score
                        fig1, fig2 = st.columns(2)
                        
                        with fig1:
                            ratio_fig = px.line(pair_df, x=pair_df.index, y='Ratio', title=f"Price Ratio: {ticker} / {pair_stock}")
                            st.plotly_chart(ratio_fig, use_container_width=True)
                            
                        with fig2:
                            zscore_fig = px.line(pair_df, x=pair_df.index, y='Ratio_Z', title="Z-Score of Price Ratio")
                            
                            # Add threshold lines for potential trading signals
                            zscore_fig.add_hline(y=2, line_dash="dash", line_color="red")
                            zscore_fig.add_hline(y=-2, line_dash="dash", line_color="green")
                            zscore_fig.add_hline(y=0, line_dash="dash", line_color="grey")
                            
                            st.plotly_chart(zscore_fig, use_container_width=True)
                        
                        # Trading signal analysis
                        st.write("#### Pair Trading Signal Analysis")
                        st.write("When the Z-score crosses above +2, it suggests the ratio is too high and may revert (sell " + ticker + ", buy " + pair_stock + ").")
                        st.write("When the Z-score crosses below -2, it suggests the ratio is too low and may revert (buy " + ticker + ", sell " + pair_stock + ").")
                        
                        # Current signal
                        current_zscore = pair_df['Ratio_Z'].iloc[-1]
                        
                        if current_zscore > 2:
                            st.write(f"**Current Signal (Z-score = {current_zscore:.2f}):** Potential short opportunity (sell {ticker}, buy {pair_stock})")
                        elif current_zscore < -2:
                            st.write(f"**Current Signal (Z-score = {current_zscore:.2f}):** Potential long opportunity (buy {ticker}, sell {pair_stock})")
                        else:
                            st.write(f"**Current Signal (Z-score = {current_zscore:.2f}):** No clear statistical arbitrage opportunity")
                    else:
                        st.error(f"Insufficient data for comparative analysis between {ticker} and {pair_stock}.")
                except Exception as e:
                    st.error(f"Error in pair trading analysis: {e}")
                    
            with adv_tab3:
                st.write("### Volatility Analysis")
                
                # Volatility over time
                st.write("#### Historical Volatility Trends")
                
                # Calculate rolling volatility for different windows
                vol_windows = [10, 20, 30, 60]
                vol_df = pd.DataFrame(index=stock_data.index)
                
                for window in vol_windows:
                    vol_df[f'{window}-Day Vol'] = stock_data['Close'].pct_change().rolling(window=window).std() * np.sqrt(252) * 100
                
                # Plot volatility
                fig = px.line(title="Historical Volatility (Annualized)")
                
                for window in vol_windows:
                    fig.add_scatter(x=vol_df.index, y=vol_df[f'{window}-Day Vol'], name=f'{window}-Day Volatility')
                
                fig.update_layout(xaxis_title="Date", yaxis_title="Annualized Volatility (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility regime analysis
                st.write("#### Volatility Regime Analysis")
                
                # Calculate median volatility to determine regimes
                median_vol = vol_df['30-Day Vol'].median()
                high_vol_threshold = median_vol * 1.5
                low_vol_threshold = median_vol * 0.5
                
                # Create volatility regimes
                vol_df['Regime'] = 'Medium Volatility'
                vol_df.loc[vol_df['30-Day Vol'] > high_vol_threshold, 'Regime'] = 'High Volatility'
                vol_df.loc[vol_df['30-Day Vol'] < low_vol_threshold, 'Regime'] = 'Low Volatility'
                
                # Calculate returns by regime
                vol_df['Return'] = stock_data['Close'].pct_change() * 100
                
                # Group by regime
                regime_returns = vol_df.groupby('Regime')['Return'].agg(['mean', 'std', 'count'])
                regime_returns['annualized_return'] = regime_returns['mean'] * 252
                regime_returns['annualized_volatility'] = regime_returns['std'] * np.sqrt(252)
                regime_returns['sharpe'] = regime_returns['annualized_return'] / regime_returns['annualized_volatility']
                
                # Display regime statistics
                st.write("#### Returns by Volatility Regime")
                
                # Format the table
                regime_table = pd.DataFrame({
                    'Volatility Regime': regime_returns.index,
                    'Annualized Return (%)': regime_returns['annualized_return'].round(2),
                    'Annualized Risk (%)': regime_returns['annualized_volatility'].round(2),
                    'Sharpe Ratio': regime_returns['sharpe'].round(2),
                    'Number of Days': regime_returns['count'].astype(int)
                })
                
                st.table(regime_table.set_index('Volatility Regime'))
                
                # Plot latest volatility
                st.write("#### Current Volatility Environment")
                
                # Get current volatility
                latest_vol = vol_df['30-Day Vol'].iloc[-1]
                
                # Create a gauge chart for volatility
                fig = go.Figure(go.Indicator(
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    value = latest_vol,
                    mode = "gauge+number+delta",
                    title = {'text': "30-Day Volatility (%)"},
                    delta = {'reference': median_vol},
                    gauge = {
                        'axis': {'range': [0, high_vol_threshold*2], 'tickwidth': 1},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, low_vol_threshold], 'color': "green"},
                            {'range': [low_vol_threshold, high_vol_threshold], 'color': "yellow"},
                            {'range': [high_vol_threshold, high_vol_threshold*2], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': high_vol_threshold
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk interpretation
                st.write("#### Volatility Interpretation")
                
                if latest_vol > high_vol_threshold:
                    st.write(f"**Current Volatility ({latest_vol:.2f}%)** is in a **high volatility regime**. Historical analysis suggests higher volatility periods for {ticker} have resulted in an average annualized return of {regime_returns.loc['High Volatility', 'annualized_return']:.2f}% with a Sharpe ratio of {regime_returns.loc['High Volatility', 'sharpe']:.2f}.")
                elif latest_vol < low_vol_threshold:
                    st.write(f"**Current Volatility ({latest_vol:.2f}%)** is in a **low volatility regime**. Historical analysis suggests lower volatility periods for {ticker} have resulted in an average annualized return of {regime_returns.loc['Low Volatility', 'annualized_return']:.2f}% with a Sharpe ratio of {regime_returns.loc['Low Volatility', 'sharpe']:.2f}.")
                else:
                    st.write(f"**Current Volatility ({latest_vol:.2f}%)** is in a **medium volatility regime**. Historical analysis suggests medium volatility periods for {ticker} have resulted in an average annualized return of {regime_returns.loc['Medium Volatility', 'annualized_return']:.2f}% with a Sharpe ratio of {regime_returns.loc['Medium Volatility', 'sharpe']:.2f}.")

    else:
        st.error(f"No data found for ticker '{ticker}'. Please check the symbol and try again.")
else:
    # Default view with tabs
    tab1, tab2 = st.tabs(["Intro", "Nifty 50 Stocks"])
    
    with tab1:
        # Default view before fetching data
        st.info("Enter a stock symbol in the sidebar and click 'Fetch Stock Data' to view financial information.")
        
        # Example of what the app can do
        st.write("""
        ### What this app can do:
        - Retrieve historical stock data for Nifty 50 companies
        - Display interactive price charts with candlestick patterns
        - Show key financial metrics and company information
        - Calculate moving averages and volatility
        - Allow downloading of all data in CSV format
        
        Enter a valid Indian stock symbol with '.NS' suffix in the sidebar (e.g., RELIANCE.NS, TCS.NS, INFY.NS) to get started.
        
        You can also select from the Nifty 50 stocks list in the next tab.
        """)
    
    with tab2:
        st.subheader("Nifty 50 Stocks")
        st.write("Click on any stock symbol to view detailed information.")
        
        # Get the Nifty 50 stocks
        nifty_stocks = get_nifty50_stocks()
        
        # Display a loading message
        with st.spinner("Loading Nifty 50 data..."):
            # Button to load all stocks (computationally expensive) or just top ones
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Top 10 Stocks (Fast)"):
                    st.session_state.load_all_nifty = False
            
            with col2:
                if st.button("Load All Nifty 50 Stocks"):
                    st.session_state.load_all_nifty = True
            
            # Initialize the session state if not exists
            if 'load_all_nifty' not in st.session_state:
                st.session_state.load_all_nifty = False
            
            # Create selections for the stocks based on user choice
            if st.session_state.load_all_nifty:
                sample_stocks = list(nifty_stocks.keys())  # Get all 50 stocks
                st.info("Loading all 50 stocks. This may take a moment...")
            else:
                sample_stocks = list(nifty_stocks.keys())[:10]  # Just get first 10 for quick loading
            
            # Get stock info
            stocks_info = get_multiple_stock_info(sample_stocks)
            
            # Create a DataFrame for display
            nifty_data = []
            for symbol, info in stocks_info.items():
                if info.get('price') != 'N/A':
                    price_str = f"₹{info['price']:,.2f}" if isinstance(info['price'], (int, float)) else "N/A"
                    change = info['change']
                    change_str = f"{change:.2f}%" if isinstance(change, (int, float)) else "N/A"
                    change_color = "green" if isinstance(change, (int, float)) and change > 0 else "red"
                    
                    market_cap = info['marketCap']
                    market_cap_str = f"₹{market_cap/1e9:,.2f}B" if isinstance(market_cap, (int, float)) else "N/A"
                    
                    pe_ratio = info['pe']
                    pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
                    
                    dividend = info['dividend']
                    dividend_str = f"{dividend*100:.2f}%" if isinstance(dividend, (int, float)) else "N/A"
                    
                    nifty_data.append({
                        "Symbol": symbol,
                        "Company": info['name'],
                        "Sector": info['sector'],
                        "Price": price_str,
                        "Change Plain": change_str, 
                        "Market Cap": market_cap_str,
                        "P/E Ratio": pe_str,
                        "Dividend Yield": dividend_str,
                        "Change":  f"<span style='color: {change_color};'>{change_str}</span>"
                    })
            
            # Convert to DataFrame
            nifty_df = pd.DataFrame(nifty_data)
            if not nifty_df.empty:
                # Add tabs for different Nifty 50 views
                nifty_tab1, nifty_tab2, nifty_tab3 = st.tabs(["Table View", "Performance Comparison", "Select & Analyze"])
                
                with nifty_tab1:
                    # Display the table with clickable symbols
                    st.write("### Nifty 50 Companies")
                    
                    # Add some filtering controls
                    filter_cols = st.columns(3)
                    with filter_cols[0]:
                        # Filter by sector
                        sectors = ["All Sectors"] + sorted(list(set([info.get('sector', 'Unknown') for info in stocks_info.values() if info.get('sector') != 'N/A'])))
                        selected_sector = st.selectbox("Filter by Sector", sectors)
                        
                    with filter_cols[1]:
                        # Sort options
                        sort_options = ["Company Name", "Market Cap (High to Low)", "Price Change (High to Low)", "P/E Ratio (Low to High)"]
                        sort_by = st.selectbox("Sort by", sort_options)
                    
                    with filter_cols[2]:
                        # Show only stocks with dividend
                        show_dividend = st.checkbox("Show only dividend paying stocks", False)
                    
                    # Apply filters
                    filtered_df = nifty_df.copy()
                    
                    # Filter by sector
                    if selected_sector != "All Sectors":
                        filtered_df = filtered_df[filtered_df["Sector"] == selected_sector]
                    
                    # Filter by dividend
                    if show_dividend:
                        filtered_df = filtered_df[filtered_df["Dividend Yield"] != "N/A" and filtered_df["Dividend Yield"] != "0.00%"]
                    
                    # Sort the dataframe
                    if sort_by == "Company Name":
                        filtered_df = filtered_df.sort_values("Company")
                    elif sort_by == "Market Cap (High to Low)":
                        # Extract numeric values for sorting
                        filtered_df["Market Cap Numeric"] = filtered_df["Market Cap"].apply(
                            lambda x: float(x.replace("₹", "").replace("B", "")) if "₹" in str(x) else 0
                        )
                        filtered_df = filtered_df.sort_values("Market Cap Numeric", ascending=False)
                        filtered_df = filtered_df.drop("Market Cap Numeric", axis=1)
                    elif sort_by == "Price Change (High to Low)":
                        # Extract numeric values for sorting (from the Change column HTML)
                        filtered_df["Change Numeric"] = filtered_df["Change"].apply(
                            lambda x: float(x.split(">")[1].split("%")[0]) if "%" in str(x) else 0
                        )
                        filtered_df = filtered_df.sort_values("Change Numeric", ascending=False)
                        filtered_df = filtered_df.drop("Change Numeric", axis=1)
                    elif sort_by == "P/E Ratio (Low to High)":
                        # Extract numeric values for sorting
                        filtered_df["PE Numeric"] = filtered_df["P/E Ratio"].apply(
                            lambda x: float(x) if x != "N/A" else float('inf')
                        )
                        filtered_df = filtered_df.sort_values("PE Numeric")
                        filtered_df = filtered_df.drop("PE Numeric", axis=1)
                    
                    # Display the filtered dataframe
                    st.dataframe(filtered_df, use_container_width=True)
                    
                    # Download option
                    st.markdown(get_csv_download_link(filtered_df, "nifty50_stocks.csv", "Download Nifty 50 Data as CSV"), unsafe_allow_html=True)
                
                with nifty_tab2:
                    st.write("### Nifty 50 Comparison Charts")
                    
                    # Create a side-by-side comparison of stocks
                    if len(nifty_data) > 1:
                        # Performance comparison - price change %
                        st.subheader("Price Change Comparison")
                        
                        # Extract symbols and price changes for the bar chart
                        symbols = []
                        changes = []
                        colors = []
                        
                        for item in nifty_data:
                            symbol = item["Symbol"].split('.')[0]  # Remove the .NS suffix for display
                            symbols.append(symbol)
                            
                            # Extract the numeric change value from the HTML string
                            change_str = item["Change"]
                            if "span" in change_str:
                                # Extract and clean the change value
                                change_value_str = change_str.split(">")[1].split("%")[0]
                                # Remove any commas before converting to float (for Indian number formats)
                                change_value = float(change_value_str.replace(",", ""))
                            else:
                                change_value = 0
                                
                            changes.append(change_value)
                            colors.append("green" if change_value >= 0 else "red")
                        
                        # Create a bar chart
                        fig_changes = go.Figure()
                        fig_changes.add_trace(go.Bar(
                            x=symbols,
                            y=changes,
                            marker_color=colors,
                            text=[f"{c:+.2f}%" for c in changes],
                            textposition="auto"
                        ))
                        
                        fig_changes.update_layout(
                            title="Price Change (%)",
                            xaxis_title="Stock",
                            yaxis_title="% Change",
                            autosize=True,
                            height=400,
                        )
                        
                        st.plotly_chart(fig_changes, use_container_width=True)
                        
                        # Market Cap comparison
                        st.subheader("Market Capitalization Comparison")
                        
                        # Extract symbols and market caps
                        market_caps = []
                        for item in nifty_data:
                            # Extract the numeric market cap value (in billions)
                            market_cap_str = item["Market Cap"]
                            if "₹" in market_cap_str:
                                # Handle Indian number format with commas
                                market_cap = float(market_cap_str.replace("₹", "").replace("B", "").replace(",", ""))
                            else:
                                market_cap = 0
                                
                            market_caps.append(market_cap)
                        
                        # Create a pie chart
                        fig_market_cap = go.Figure(data=[go.Pie(
                            labels=symbols,
                            values=market_caps,
                            hole=.4,
                            textinfo="label+percent"
                        )])
                        
                        fig_market_cap.update_layout(
                            title="Market Cap Distribution",
                            autosize=True,
                            height=500,
                        )
                        
                        st.plotly_chart(fig_market_cap, use_container_width=True)
                        
                        # Sector breakdown
                        st.subheader("Sector Distribution")
                        
                        # Group by sector
                        sectors = {}
                        for item in nifty_data:
                            sector = item["Sector"]
                            if sector in sectors:
                                sectors[sector] += 1
                            else:
                                sectors[sector] = 1
                        
                        # Create a bar chart for sectors
                        fig_sectors = go.Figure(data=[go.Bar(
                            x=list(sectors.keys()),
                            y=list(sectors.values()),
                            marker_color="lightblue",
                            text=list(sectors.values()),
                            textposition="auto"
                        )])
                        
                        fig_sectors.update_layout(
                            title="Number of Companies by Sector",
                            xaxis_title="Sector",
                            yaxis_title="Count",
                            autosize=True,
                            height=400,
                        )
                        
                        st.plotly_chart(fig_sectors, use_container_width=True)
                        
                    else:
                        st.info("Load more stocks to see comparison charts")
                
                with nifty_tab3:
                    # Display list of all 50 stocks in columns
                    st.write("### All Nifty 50 Stocks")
                    
                    # Create columns for the full list
                    cols = st.columns(3)
                    for i, (symbol, company) in enumerate(nifty_stocks.items()):
                        col_idx = i % 3
                        cols[col_idx].write(f"**{symbol}**: {company}")
                    
                    # Market trend information
                    st.subheader("Select a Nifty 50 Stock to Analyze")
                    selected_nifty = st.selectbox("Choose a stock", list(nifty_stocks.keys()), format_func=lambda x: f"{x} - {nifty_stocks[x]}")
                    
                    # Get time range for analysis
                    time_range_cols = st.columns(3)
                    with time_range_cols[0]:
                        if st.button("6 Months"):
                            start_date_for_selected = datetime.now() - timedelta(days=180)
                            st.session_state.custom_nifty_start_date = start_date_for_selected
                    
                    with time_range_cols[1]:
                        if st.button("1 Year"):
                            start_date_for_selected = datetime.now() - timedelta(days=365)
                            st.session_state.custom_nifty_start_date = start_date_for_selected
                    
                    with time_range_cols[2]:
                        if st.button("3 Years"):
                            start_date_for_selected = datetime.now() - timedelta(days=365*3)
                            st.session_state.custom_nifty_start_date = start_date_for_selected
                    
                    # If time range selected, use it
                    if 'custom_nifty_start_date' in st.session_state:
                        custom_message = f"Analyzing from {st.session_state.custom_nifty_start_date.strftime('%d %b %Y')} to today"
                        st.info(custom_message)
                    
                    if st.button("Analyze Selected Nifty Stock"):
                        # Set the selected stock in session state
                        st.session_state.temp_ticker = selected_nifty
                        
                        # Set custom date if available
                        if 'custom_nifty_start_date' in st.session_state:
                            st.session_state.start_date = st.session_state.custom_nifty_start_date
                        
                        # Rerun to use the selection
                        st.rerun()
            else:
                st.error("Error loading Nifty 50 data. Please check your internet connection and try again.")

# Footer
st.markdown("---")
st.markdown("Stock data provided by Yahoo Finance via yfinance")
