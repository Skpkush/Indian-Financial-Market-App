# Indian-Financial-Market-App

### Indian-Financial-Market-App


Screeenshot
![image](https://github.com/user-attachments/assets/ed59cc71-9399-4b2c-8be8-467f31db1a89)


This **Streamlit-based web application**  empowers users to access, analyze, and visualize financial data for Nifty 50 stocks, sourced from the Indian Stock Market. Designed for investors, traders, and data enthusiasts, the app provides tools for interactive exploration of market data. This platform is more than just a data viewer; it's a dynamic hub for financial insights, tailored to meet the diverse needs of its users. Whether you're a seasoned investor meticulously tracking market trends, a day trader seeking real-time analytics, or a student eager to learn the intricacies of the stock market, this application provides the resources and functionalities you need. By providing a user-friendly interface and a comprehensive suite of tools, the app aims to democratize access to financial information, enabling users to make informed decisions and navigate the complexities of the Indian stock market with confidence.

## **Key features include**:

1. **Historical Stock Data:** Retrieval of historical stock data for Nifty 50 companies. This feature allows users to delve into the past performance of Nifty 50 stocks, gaining insights into long-term trends, identifying patterns, and understanding how these stocks have reacted to various market conditions over time. The data can be used to inform investment strategies and risk assessment.

2. **Interactive Charts:** Display of interactive price charts with candlestick patterns. These interactive charts provide a dynamic and user-friendly way to visualize stock price movements. Candlestick patterns, in particular, offer valuable information about market sentiment, potential reversals, and price trends. Users can zoom in on specific time periods, analyze trading volumes, and gain a deeper understanding of market dynamics.

3. **Financial Metrics:** Presentation of key financial metrics such as market cap, P/E ratio, and dividend yield. The app provides a snapshot of a company's financial health by presenting key metrics. Market capitalization indicates the total value of a company's outstanding shares, the P/E ratio reflects the relationship between a company's stock price and its earnings, and the dividend yield shows the return on investment from dividends. These metrics help investors assess a company's valuation, profitability, and income potential.

4. **Technical Analysis:**Calculation of moving averages, volatility, and other technical indicators. The application includes tools for technical analysis, enabling users to identify trends and potential trading opportunities. Moving averages smooth out price data to highlight longer-term trends, while volatility measures the degree of price fluctuations. Other indicators may include RSI, MACD, etc., which provide further insights into market momentum, overbought/oversold conditions, and possible trend reversals.

5.** Data Export:** Download of all data in CSV format for further analysis. This feature enables users to export the data for use in other applications or for more in-depth, customized analysis. The CSV format is widely compatible with spreadsheet software and programming languages, providing flexibility for users to conduct their own research, create custom visualizations, or integrate the data into existing workflows.

User-Friendly Interface: A simple and intuitive interface powered by Streamlit. The Streamlit framework allows for the creation of a user-friendly interface, making the app accessible to both novice and experienced users. The intuitive design ensures that users can easily navigate the app, access the data they need, and utilize the various features without a steep learning curve.


## **How to Use**
1. **Select a Stock**:
   - Choose a Nifty 50 stock from the dropdown list in the sidebar.
   - Alternatively, enter a custom stock symbol (e.g., `RELIANCE.NS`, `TCS.NS`).

2. **View Data**:
   - The app will display:
     - Historical price data.
     - Interactive candlestick charts.
     - Key financial metrics (e.g., market cap, P/E ratio, dividend yield).

3. **Download Data**:
   - Click the "Download CSV" button to export the data for offline analysis.

---

## **Installation**
Follow these steps to set up the project locally on your machine:

### 1. **Clone the Repository**
   - Open your terminal or command prompt.
   - Run the following command to clone the repository to your local machine:
     ```bash
     git clone https://github.com/your-username/Indian-Financial-Market-App.git
     ```
     Replace `your-username` with your actual GitHub username.
   - Navigate into the project directory:
     ```bash
     cd Indian-Financial-Market-App
     ```

### 2. **Install Dependencies**
   - Install all the required Python libraries listed in the `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```
   - This will install the following dependencies:
     - `streamlit`: For building the web app.
     - `yfinance`: For fetching stock data from Yahoo Finance.
     - `pandas`: For data manipulation and analysis.
     - `numpy`: For numerical computations.
     - `plotly`: For interactive data visualization.
     - `ta`: For technical analysis calculations.
     - `nsetools`: For accessing NSE (National Stock Exchange) data.
     - `scipy`: For statistical calculations.

### 3. **Run the Application**
   - Start the Streamlit app by running:
     ```bash
     streamlit run app.py
     ```
   - This will start a local server, and you’ll see an output similar to:
     ```
     You can now view your Streamlit app in your browser.
     Local URL: http://localhost:8501
     Network URL: http://192.168.x.x:8501
     ```

### 4. **Access the App**
   - Open your web browser and go to the URL provided in the terminal (usually `http://localhost:8501`).
   - You should now see the **Indian Financial Market App** running in your browser.

---

## **Dependencies**
The project uses the following Python libraries:
- `streamlit`: For building the web app.
- `yfinance`: For fetching stock data from Yahoo Finance.
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `plotly`: For interactive data visualization.
- `ta`: For technical analysis calculations.
- `nsetools`: For accessing NSE (National Stock Exchange) data.
- `scipy`: For statistical calculations.

All dependencies are listed in the `requirements.txt` file.

---




