from os import stat
from attr import attrib
import streamlit as st
from streamlit.type_util import Key
import yfinance as yf
import pandas as pd
import yahoo_fin.stock_info as si
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as pdr


# main date function to compute for intervals
def date_function(n):
    from dateutil.relativedelta import relativedelta
    current_date = datetime.today()
    print('Current Date: ', current_date)
    timeframe = 0
    if(n == '1M'):
        timeframe = 1
    elif(n == '2M'):
        timeframe = 2
    elif(n == '3M'):
        timeframe = 3
    elif(n == '6M'):
        timeframe = 6
    elif(n == 'YTD'):
        timeframe = datetime.now().month
    elif(n == '1Y'):
        timeframe = 12
    elif(n == '3Y'):
        timeframe = 36
    elif(n == '5Y'):
        timeframe = 60
    past_date = current_date - relativedelta(months=timeframe)
    # Convert datetime object to string in required format
    date_format = '%Y/%m/%d'
    past_date_str = past_date.strftime(date_format)
    print('Date (as string) - 20 months before current date: ', past_date_str)
    return past_date_str


# set default ticker to sp500
ticker_list = ['-'] + si.tickers_sp500()
# create ticker items dropdown
ticker_item = st.sidebar.selectbox(
    'Pick your assets', ticker_list, key="menu_sidebar")

# duration of time
range_array = ['1M', '2M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX']
# create sidebar dropdown for duration
period = st.sidebar.selectbox('Period', range_array, key="time-period-tab1")


def tab1():
    # Add dashboard title and description
    st.title("YAHOO FINANCIAL DASHBOARD")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 1 - COMPANY PROFILE')

    @st.cache
    def GetCompanyInfo(ticker_item):
        return si.get_company_info(ticker_item)

    # show warning when stock isn't selected
    if(ticker_item == '-'):
        st.warning('Select a stock to see data')

    if ticker_item != '-':
        info = GetCompanyInfo(ticker_item)
        info['Value'] = info['Value'].astype(str)
        st.dataframe(info, height=2000)


def tab2():
    st.header('Summary')
    st.write('***************************************************************')
    # set sidebar header
    st.sidebar.header('Select Tickers to Get Started')


    # Add table to show stock data

    @st.cache
    def GetCompanyInfo(ticker_item):
        return si.get_quote_table(ticker_item)

    # show warning when stock isn't selected
    if(ticker_item == '-'):
        st.warning('Select a stock to see data')

    if ticker_item != '-':
        info = GetCompanyInfo(ticker_item)
        data = pd.DataFrame.from_dict(info, orient='index').astype(str)
        st.dataframe(data, height=2000)

    start = pd.to_datetime(date_function(period))
    today = pd.to_datetime('today')

    # stock chart
    df = yf.download(ticker_item, start, today)
    st.line_chart(df)

    # update button to download and update the data
    st.sidebar.write('Select an option to either download or update')

    options = ['-', 'update', 'download']
    selected_option = st.sidebar.selectbox('Select an action', options)

    # download dataframe as csv after adding filename (filename optional. it would result .default.csv if it's not present)
    def download_as_csv(filename):
        if not filename:
            df.to_csv('default.csv', index=False)
        else:
            df.to_csv(filename + '.csv', index=False)

    if(str(selected_option).lower() == 'update'):
        st.sidebar.button(selected_option, on_click=None, key=None)
    elif(str(selected_option).lower() == 'download'):
        label = 'Enter name of file here'
        download_file_value = st.text_input(label, value="", max_chars=None, key=None, type="default",
                                            help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None)
        st.button(selected_option, on_click=download_as_csv(
            download_file_value), key="1_download_function")
    else:
        pass


def tab3():
    st.header('Chart')
    st.write('***************************************************************')

    # Reference of Code https://asxportfolio.com/shares-python-for-finance-plotting-stock-data
    start = pd.to_datetime(date_function(period))
    today = pd.to_datetime('today')

    # Add table to show stock data
    @st.cache
    def GetStockData(tickers, start, today):
        return pd.concat([si.get_data(tick, start, today) for tick in tickers])
    stock_price = GetStockData([ticker_item], start, today)

    # show warning when stock isn't selected
    if(ticker_item == '-'):
        st.warning('Select a stock to see data')

    if ticker_item != '-':
        stock_price = GetStockData([ticker_item], start, today)

    stock_price['MA50'] = stock_price['close'].rolling(
        window=50, min_periods=0).mean()
    stock_price['MA200'] = stock_price['close'].rolling(
        window=200, min_periods=0).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.10, subplot_titles=('ticker', 'volume'),
                        row_width=[0.2, 0.7])

    fig.add_trace(go.Candlestick(x=stock_price.index, open=stock_price["open"], high=stock_price["high"],
                                 low=stock_price["low"], close=stock_price["close"], name="OHLC"),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=stock_price.index,
                  y=stock_price["MA50"], marker_color='grey', name="MA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_price.index,
                  y=stock_price["MA200"], marker_color='lightgrey', name="MA200"), row=1, col=1)

    fig.add_trace(go.Bar(x=stock_price.index,
                  y=stock_price['volume'], marker_color='red', showlegend=False), row=2, col=1)

    fig.update_layout(
        title='Stock Historical price chart',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Price ($/share)',
            titlefont_size=14,
            tickfont_size=12,
        ),
        autosize=False,
        width=800,
        height=500,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        paper_bgcolor='White'
    )

    fig.update(layout_xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)


def tab4():
    st.header('Statistics')
    st.write('***************************************************************')

    # show warning when stock isn't selected
    if(ticker_item == '-'):
        st.warning('Select a stock to see data')

    # Add table to show statistics data

    @st.cache
    def GetStatistics(ticker):
        return si.get_stats_valuation(ticker)
    
    @st.cache
    def GetStat(ticker):
        return si.get_stats(ticker)

    col1, col2 = st.columns(2)

    with col1:
        if ticker_item != '-':
            st.write('Valuation Measures')
            Valuation_Measures = GetStatistics(ticker_item)
            st.dataframe(Valuation_Measures, height=1000)
            
        if ticker_item != '-':
            st.header('Financial Highlight')
            st.write('Fiscal Year')
            Fiscal_Year = GetStat(ticker_item).iloc[29:31]
            st.dataframe(Fiscal_Year, height=1000)
            
        if ticker_item != '-':
            st.write('Profitability')
            Profitability = GetStat(ticker_item).iloc[31:33]
            st.dataframe(Profitability, height=1000)
        
        if ticker_item != '-':
            st.write('Management Effectiveness')
            Management_Effectiveness = GetStat(ticker_item).iloc[33:35]
            st.dataframe(Management_Effectiveness, height=1000)
            
        if ticker_item != '-':
            st.write('Income Statement')
            Income_Statement = GetStat(ticker_item).iloc[35:43]
            st.dataframe(Income_Statement, height=1000)
            
        if ticker_item != '-':
             st.write('Balance Sheet')
             Balance_Sheet = GetStat(ticker_item).iloc[43:50]
             st.dataframe(Balance_Sheet, height=1000)

        if ticker_item != '-':
             st.write('Cash Flow Statement')
             Cash_Flow_Statement = GetStat(ticker_item).iloc[50:]
             st.dataframe(Cash_Flow_Statement, height=1000)
             
            
    with col2:
        
        if ticker_item != '-':
            st.header('Trading Information')
            st.write('Stock Price History')
            Stock_Price_History = GetStatistics(ticker_item).iloc[:7]
            st.dataframe(Stock_Price_History, height=1000)
            
        if ticker_item != '-':
            st.write('Share Statistics')
            Share_Statistics = GetStat(ticker_item).iloc[7:19]
            st.dataframe(Share_Statistics, height=1000)
    
        if ticker_item != '-':
          st.write('Dividends and Splits')
          Dividends_Splits = GetStat(ticker_item).iloc[19:29]
          st.dataframe(Dividends_Splits, height=1000)


def tab5():
    st.header('Financials')
    st.write('***************************************************************')

    col1, col2 = st.columns(2)

    with col1:
        option = st.selectbox('Select View Options', [
                              'General View', 'Income Statement', 'Balance Sheet', 'Cash Flow'], key="financial_view_options")
    with col2:
        time_option = st.selectbox('Select Display Interval', [
                                   'Annual', 'Quarterly'], key="time_intervals_to_show_financials")

    # show warning when stock isn't selected
    if(ticker_item == '-'):
        st.warning('Select a stock to see data')

    def option_to_lower():
        return option.lower()

    if(ticker_item != '-'):
        if(time_option == 'Annual'):
            if(option_to_lower() == 'income statement'):
                st.write('Yearly Income Statement')
                financials = si.get_financials(ticker_item)
                st.dataframe(
                    financials['yearly_income_statement'], height=2000)

            elif(option_to_lower() == 'balance sheet'):
                st.write('Yearly Balance Sheet')
                financials = si.get_financials(ticker_item)
                st.dataframe(financials['yearly_balance_sheet'], height=2000)

            elif(option_to_lower() == 'cash flow'):
                st.write('Yearly Cash Flow')
                financials = si.get_financials(ticker_item)
                st.dataframe(financials['yearly_cash_flow'], height=2000)

            elif(option_to_lower() == 'general view'):
                st.header('Yearly Financial Data')
                financials = si.get_financials(ticker_item)
                st.write('Yearly Income Statement')
                st.dataframe(
                    financials['yearly_income_statement'], height=2000)
                st.write('Yearly Balance Sheet')
                st.dataframe(financials['yearly_balance_sheet'], height=2000)
                st.write('Yearly Cash Flow')
                st.dataframe(financials['yearly_cash_flow'], height=2000)
        if(time_option == 'Quarterly'):
            if(option_to_lower() == 'income statement'):
                st.write('Quarterly Income Statement')
                financials = si.get_financials(ticker_item)
                st.dataframe(
                    financials['quarterly_income_statement'], height=2000)

            elif(option_to_lower() == 'balance sheet'):
                st.write('Quarterly Balance Sheet')
                financials = si.get_financials(ticker_item)
                st.dataframe(
                    financials['quarterly_balance_sheet'], height=2000)

            elif(option_to_lower() == 'cash flow'):
                st.write('Quarterly Cash Flow')
                financials = si.get_financials(ticker_item)
                st.dataframe(financials['quarterly_cash_flow'], height=2000)

            elif(option_to_lower() == 'general view'):
                st.header('Quarterly Financial Data')
                financials = si.get_financials(ticker_item)
                st.write('Quarterly Income Statement')
                st.dataframe(
                    financials['quarterly_income_statement'], height=2000)
                st.write('Quarterly Balance Sheet')
                st.dataframe(
                    financials['quarterly_balance_sheet'], height=2000)
                st.write('Quarterly Cash Flow')
                st.dataframe(financials['quarterly_cash_flow'], height=2000)


def tab7():
     st.header('Monte Carlo Simulation')
     st.write('***************************************************************')


     # option button to simulate the data
     st.sidebar.write('Select an option for either number of simulations or time frame')

     options = ['-','200', '500', '1000']
     number_of_simulation = st.sidebar.selectbox('Select number of simulation', options)

     option = ['-','30', '60', '90']
     n_simulation = st.sidebar.selectbox('Select a time horizon', option)

     start = pd.to_datetime(date_function(period))
     today = pd.to_datetime('today')


     class MonteCarlo(object):

         def __init__(self, ticker_item, data_source, start, today, time_horizon, n_simulation, seed):

             # Initiate class variables
             self.ticker = ticker_item  # Stock ticker
             self.data_source = data_source  # Source of data, e.g. 'yahoo'
             self.start = start  # Text, YYYY-MM-DD
             self.today = today  # Text, YYYY-MM-DD
             self.time_horizon = time_horizon  # Days
             self.n_simulation = n_simulation  # Number of simulations
             self.seed = seed  # Random seed
             self.simulation_df = pd.DataFrame()  # Table of results


             # Extract stock data
             self.stock_price = pdr.DataReader(ticker_item, data_source, self.start, self.today)

             # Calculate financial metrics
             # Daily return (of close price)
             self.daily_return = self.stock_price['Close'].pct_change()
             # Volatility (of close price)
             self.daily_volatility = np.std(self.daily_return)


         def run_simulation(self):

             # Run the simulation
             np.random.seed(self.seed)
             self.simulation_df = pd.DataFrame()  # Reset

             for i in range(self.n_simulation):

                 # The list to store the next stock price
                 next_price = []

                 # Create the next stock price
                 last_price = self.stock_price['Close'][-1]

                 for j in range(self.time_horizon):

                     # Generate the random percentage change around the mean (0) and std (daily_volatility)
                     future_return = np.random.normal(0, self.daily_volatility)

                     # Generate the random future price
                     future_price = last_price * (1 + future_return)

                     # Save the price and go next
                     next_price.append(future_price)
                     last_price = future_price

                 # Store the result of the simulation
                 self.simulation_df[i] = next_price

         def plot_simulation_price(self):

             # Plot the simulation stock price in the future
             fig, ax = plt.subplots()
             fig.set_size_inches(15, 10, forward=True)

             plt.plot(self.simulation_df)
             plt.title('Monte Carlo simulation for ' + self.ticker + \
                       ' stock price in next ' + str(self.time_horizon) + ' days')
             plt.xlabel('Day')
             plt.ylabel('Price')

             plt.axhline(y=self.stock_price['Close'][-1], color='red')
             plt.legend(['Current stock price is: ' + str(np.round(self.stock_price['Close'][-1], 2))])
             ax.get_legend().legendHandles[0].set_color('red')

             st.pyplot(fig)

         def plot_simulation_hist(self):

             # Get the ending price of the 200th day
             ending_price = self.simulation_df.iloc[-1:, :].values[0, ]

             # Plot using histogram
             plt.hist(ending_price, bins=50)
             plt.axvline(x=self.stock_price['Close'][-1], color='red')
             plt.legend(['Current stock price is: ' + str(np.round(self.stock_price['Close'][-1], 2))])
             #ax.get_legend().legendHandles[0].set_color('green')
             plt.show()

         def value_at_risk(self):
             # Price at 95% confidence interval
             future_price_95ci = np.percentile(self.simulation_df.iloc[-1:, :].values[0, ], 5)

             # Value at Risk
             VaR = self.stock_price['Close'][-1] - future_price_95ci
             print('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')

     # Initiate
     data_source ='yahoo'
     mc_sim = MonteCarlo(ticker_item, data_source, start, today, time_horizon=30, n_simulation=1000, seed=123)

     # Run simulation
     mc_sim.run_simulation()

     # Plot the results
     mc_sim.plot_simulation_price()

     # Plot the results
     mc_sim.plot_simulation_hist()

     # Value at risk
     mc_sim.value_at_risk()


def tab6():
    st.header('Analysis')
    st.write('***************************************************************')
  
    start = pd.to_datetime(date_function(period))
    today = pd.to_datetime('today')

    @st.cache
    def GetStockData(ticker, start, today):
     return si.get_analysts_info(ticker)
 
    # show warning when stock isn't selected
    if(ticker_item == '-'):
        st.warning('Select a stock to see data')
        
   # col1 = st.columns(1)

    #with col1:
        
    if ticker_item != '-':
         st.write('Earnings Estimate')
         Earnings_Estimate = GetStockData(ticker_item, start, today)
         print(Earnings_Estimate)
         #data = pd.DataFrame.from_dict(Revenue_Estimate, orient='index').astype(str)
         st.dataframe(Earnings_Estimate['Earnings Estimate'], height=2000)
         
    if ticker_item != '-':
         st.write('Revenue Estimate')
         Revenue_Estimate = GetStockData(ticker_item, start, today)
         print(Revenue_Estimate)
         #data = pd.DataFrame.from_dict(Revenue_Estimate, orient='index').astype(str)
         st.dataframe(Revenue_Estimate['Revenue Estimate'], height=2000)
         
    if ticker_item != '-':
       st.write('Earnings History')
       Earnings_History = GetStockData(ticker_item, start, today)
       print(Earnings_History)
       #data = pd.DataFrame.from_dict(Revenue_Estimate, orient='index').astype(str)
       st.dataframe(Earnings_History['Earnings History'], height=2000)   
       
    if ticker_item != '-':
         st.write('EPS Trend')
         EPS_Trend = GetStockData(ticker_item, start, today)
         print(EPS_Trend)
         #data = pd.DataFrame.from_dict(Revenue_Estimate, orient='index').astype(str)
         st.dataframe(EPS_Trend['EPS Trend'], height=2000)
    
    if ticker_item != '-':
         st.write('EPS Revisions')
         EPS_Revisions = GetStockData(ticker_item, start, today)
         print(EPS_Revisions)
         #data = pd.DataFrame.from_dict(Revenue_Estimate, orient='index').astype(str)
         st.dataframe(EPS_Revisions['EPS Revisions'], height=2000)
    
    if ticker_item != '-':
         st.write('Growth Estimates')
         Growth_Estimates = GetStockData(ticker_item, start, today)
         print(Growth_Estimates)
         #data = pd.DataFrame.from_dict(Revenue_Estimate, orient='index').astype(str)
         st.dataframe(Growth_Estimates['Growth Estimates'], height=2000)
         

         
def tab8():
    st.header('Total Traded Analysis')
    st.write('***************************************************************')

    # Reference of code https://www.youtube.com/watch?v=57qAxRV577c&t=267s
    start = pd.to_datetime(date_function(period))
    today = pd.to_datetime('today')

    # Add table to show stock data
    @st.cache
    def GetStockData(tickers, start, today):
        return pd.concat([si.get_data(tick, start, today) for tick in tickers])
    
    stock_price = GetStockData([ticker_item], start, today)
    stock_price['Total Traded'] = stock_price['volume'] * stock_price['open']

    # Add a line plot
    fig, ax = plt.subplots(figsize=(15, 7))
    stock_df = stock_price['Total Traded']
    ax.plot(stock_df, label=ticker_item, alpha=1.0, color='yellow')
    ax.set_title('Total Traded Analysis Chart')
    ax.set_xlabel('Period')
    ax.set_ylabel('Total Traded')
    ax.legend()
    st.pyplot(fig)


def run():
    # Add a radio box
    select_tab = st.sidebar.radio(
        "Select tab", ['Company profile', 'Summary', 'Chart', 'Statistics', 'Financials', 'Analysis', 'Monte Carlo Simulation', 'Total Traded Analysis'])

    # Show the selected tab
    if select_tab == 'Company profile':
        tab1()
    elif select_tab == 'Summary':
        tab2()
    elif select_tab == 'Chart':
        tab3()
    elif select_tab == 'Statistics':
        tab4()
    elif select_tab == 'Financials':
        tab5()
    elif select_tab == 'Analysis':
        tab6()
    elif select_tab == 'Monte Carlo Simulation':
         tab7()
    elif select_tab == 'Total Traded Analysis':
        tab8()


if __name__ == "__main__":
    run()
