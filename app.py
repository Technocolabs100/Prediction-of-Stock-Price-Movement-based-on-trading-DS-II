############################Libraries_Used#############

import streamlit as st

import yfinance as yf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import urllib

from keras.models import load_model
# import Image from pillow to open images
from PIL import Image
import pickle

import plotly.express as px
import datetime as dt
from streamlit import caching
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def user_input_features1():
    date = st.date_input('Select Date')
    date = date.day
    hour = st.slider('Hour of the day', 9, 17, 9)
    minute = st.slider('Minute of the day', 0, 59, 5)

    return date, hour, minute


def user_input_features2():
    date = st.date_input('Select Date')
    date = date.day
    return date


def calcMovingAverage(data, size):
    df = data.copy()
    df['sma'] = df['open'].rolling(size).mean()
    df['ema'] = df['open'].ewm(span=size, min_periods=size).mean()
    df.dropna(inplace=True)
    return df



@st.cache
def load_data(nrows):
    data = pd.read_csv(r'test.csv',
                       nrows=nrows)
    return data


@st.cache
def load_msft_data(nrows):
    data = pd.read_csv(r'MSFT.csv',
                       nrows=nrows)
   
    return data


@st.cache
def load_nfty_data(nrows):
    data = pd.read_csv(r'DataFrame.csv',
                       nrows=nrows)
   
    return data





st.markdown(

            "<center><h1 style='text-align: centre; color: black;'>Prediction-of-Stock-Price-Movement-based-on-trading-DS-II</h1></center>",
            unsafe_allow_html=True)


#############################################dataset and raw data
my_expander6 = st.beta_expander("Get fundamentals by choosing Stocks(SNP 500)", expanded=False)
with my_expander6:
   
    snp500 = pd.read_csv(r'SP500.csv')
    symbols = snp500['Symbol'].sort_values().tolist()

    ticker = st.selectbox(
        'Choose a S&P 500 Stock',
        symbols)
    stock = yf.Ticker(ticker)
    info = stock.info
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    infoType = st.radio(
        "Choose an info type",
        ('Company Profile', 'Market trend', 'Fundamental Info', 'Market Info', 'Business Summary')
    )
    if infoType == 'Company Profile':

        st.title('Company Profile')
        st.subheader(info['longName'])

        col01, col02, col03 = st.beta_columns(3)

        
        sector = info['sector']
        col01.header("Sector")
        col01.write(sector)

       
        industry = info['industry']
        
        col02.header("Industry")
        col02.write(industry)

        website = info['website']
        col03.header("website")
        col03.write(website)

        col04, col05, col06 = st.beta_columns(3)

        market = info['market']
        col04.header("Market")
        col04.write(market)

        
        quo = info['quoteType']
        col06.header("quote type")
        col06.write(quo)

        exch = info['exchange']
        col05.header("Exchange")
        col05.write(exch)

    elif infoType == 'Business Summary':
        st.markdown('** Business Summary **')
        st.info(info['longBusinessSummary'])

    elif infoType == 'Fundamental Info':
        fundInfo = {
            'Enterprise Value (USD)': info['enterpriseValue'],
            'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
            'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
            'Net Income (USD)': info['netIncomeToCommon'],
            'Profit Margin Ratio': info['profitMargins'],
            'Forward PE Ratio': info['forwardPE'],
            'PEG Ratio': info['pegRatio'],
            'Price to Book Ratio': info['priceToBook'],
            'Forward EPS (USD)': info['forwardEps'],
            'Beta ': info['beta'],
            'Book Value (USD)': info['bookValue'],
            'Dividend Rate (%)': info['dividendRate'],
            'Dividend Yield (%)': info['dividendYield'],
            'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
            'Payout Ratio': info['payoutRatio']
        }

        fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
        fundDF = fundDF.rename(columns={0: 'Value'})
        st.subheader('Fundamental Info')
        st.table(fundDF)
    elif infoType == 'Market Info':
        marketInfo = {
            "Volume": info['volume'],
           
            "Market Cap": info["marketCap"],
            "Float Shares": info['floatShares'],
            'Bid Size': info['bidSize'],
            'Ask Size': info['askSize'],
            "Share Short": info['sharesShort'],
            "Regular Market Price (USD)": info['regularMarketPrice']
            
        }

        marketDF = pd.DataFrame(data=marketInfo, index=[0])
        st.table(marketDF)
    else:
        start = dt.datetime.today() - dt.timedelta(2 * 365)
        end = dt.datetime.today()
        df = yf.download(ticker, start, end)
        df = df.reset_index()
        fig = go.Figure(
            data=go.Scatter(x=df['Date'], y=df['Adj Close'])
        )
        fig.update_layout(
            title={
                'text': "Stock Prices Over Past Two Years",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig, use_container_width=True)


def show_info(df):
    
    col1, col2 = st.beta_columns(2)

    original = df.open[0]
    col1.header("Open Price")
    col1.write(original)

   
    grayscale = df.close[0]
    col2.header("Prev Close")
    col2.write(grayscale)


    st.write("last day return")
    x = 0
    col4, col5, col6, col7 = st.beta_columns(4)
    return1 = (df.open[x] - df.close[0]) / (df.open[x]) * 100
    fo = "{:.2f}".format(return1)
    col4.header("Profit % last day")
    col4.write(fo)

    return2 = (df.open[x] - df.close[6]) / (df.open[x]) * 100
    fo2 = "{:.2f}".format(return2)
    col5.header("Profit % last week")
    col5.write(fo2)

    return3 = (df.open[x] - df.close[30]) / (df.open[x]) * 100
    fo3 = "{:.2f}".format(return3)
    col6.header("Profit % last month")
    col6.write(fo3)

    return4 = (df.open[x] - df.close[365]) / (df.open[x]) * 100
    fo4 = "{:.2f}".format(return4)
    col7.header("Profit % last year")
    col7.write(fo4)


my_expander7 = st.beta_expander("Basic Info ", expanded=False)
with my_expander7:
    st.header('Select Dataset to view info')
    df = pd.DataFrame({
        'first column': ["MSFT", "NIFTY"]
    })
    option1 = st.selectbox(
        'Select prediction Interval',
        df['first column'])

    if option1 == "MSFT":
        #'You selected:', option1

        df1 = pd.read_csv(r'test_002.csv')
        df1.columns = [x.lower() for x in df1.columns]
       
        show_info(df1)
    if option1 == "NIFTY":
        #'You selected:', option1
        df2 = pd.read_csv(
            r'DataFrame.csv')
        
        show_info(df2)

my_expander2 = st.beta_expander("Plotting Visualization", expanded=False)
with my_expander2:


    # extracting Data for plotting

    df = load_msft_data(8858)
   
    df.columns = map(str.lower, df.columns)
    df = df[['date', 'open', 'high',
             'low', 'close', 'volume']]
   
    df.set_index(pd.DatetimeIndex(df['date']), inplace=True)
   
    st.write("Customize by selecting the area")
    fig4 = go.Figure(data=[go.Candlestick(x=df['date'],
                                          open=df['open'], high=df['high'],
                                          low=df['low'], close=df['close'])
                           ])

    fig4.update_layout(xaxis_rangeslider_visible=False)
    # fig4.show()
    st.write(fig4)



my_expander3 = st.beta_expander("Visualization with Indicators", expanded=False)
with my_expander3:
    df = pd.read_csv(r'test_002.csv')

    df['MA12'] = df['open'].rolling(12).mean()
    df['VMA12'] = df['volume'].rolling(12).mean()
    genre = st.radio(
        "12 days Moving Average with",
        ('Open', 'Volume', 'Custom'))
    if genre == 'Open':
        st.write('Open with 12days moving average ')
        df_melt = df.melt(id_vars='date', value_vars=["MA12", "open"])
        fig1 = px.line(df_melt, x="date", y='value', template='plotly_dark', color="variable")
        st.write(fig1)

    elif genre == 'Volume':
        st.write("Volume with 12days moving Average ")
        df_melt = df.melt(id_vars='date', value_vars=["VMA12", "volume"])
        fig2 = px.line(df_melt, x="date", y='value', template='plotly_dark', color="variable")
        st.write(fig2)

    elif genre == 'Custom':
      
        coMA1, coMA2 = st.beta_columns(2)

        with coMA1:
            numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)

        with coMA2:
            windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)

        start = dt.datetime.today() - dt.timedelta(numYearMA * 365)
        end = dt.datetime.today()
        
        df_ma = calcMovingAverage(df, windowSizeMA)
        df_ma = df_ma.reset_index()
        figMA = go.Figure()

        figMA.add_trace(
            go.Scatter(
                x=df_ma['date'],
                y=df_ma['open'],
                name="Prices Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x=df_ma['date'],
                y=df_ma['sma'],
                name="SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x=df_ma['date'],
                y=df_ma['ema'],
                name="EMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))

        figMA.update_layout(legend_title_text='Trend')
        figMA.update_yaxes(tickprefix="$")

        st.plotly_chart(figMA, use_container_width=True)

my_expander4 = st.beta_expander("Visualization and Prediction(on specific dataset)", expanded=False)
with my_expander4:
    st.header('Users Input Parameters')
    df = pd.DataFrame({
        'first column': ["MSFT 1- Day", "NIFTY 1-Minute"]
    })
    option = st.selectbox(
        'Select prediction Interval',
        df['first column'])

    if option == "NIFTY 1-Minute":
       
        df = pickle.load(open('df_nifty.pkl', 'rb'))
        
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        model = load_model('model.h5')
        with open('ftest.pkl', 'rb') as f:
            f_test = pickle.load(f)

        f_test = np.array(f_test)
        f_test = np.reshape(f_test, (f_test.shape[0], f_test.shape[1], 1))

        st.header('User Input Parameters')

        day, hour, minute = user_input_features1()

        f_predict = []
        n_days = 1
        minutes = 0
        hours = 8  # [9--17]

        for i in range(n_days):
            while (hours > 0):
                while (minutes < 60):
                    res = model.predict(f_test)
                    f_predict.append(res[0][0])
                    f_test = np.delete(f_test, [0], None)
                    f_test = np.append(f_test, res[0][0], None)
                    f_test = f_test.reshape(1, 60, 1)
                    minutes = minutes + 1
                hours = hours - 1
                minutes = 0
            hours = 8
            minutes = 0

        hour = hour - 8

        st.header("Prediction:")
        res = scaler.inverse_transform([[f_predict[day * hour * minute]]])

        col1, col2, col3 = st.beta_columns(3)

        original = df.open[0]
        col1.subheader("Open Price")
        col1.write(res[0][0])

        grayscale = df.open[0]
        col2.subheader("Prev Day Open")
        a = scaler.inverse_transform([[f_predict[day - 2]]])[0][0]
        col2.write(a)

        return1 = (a - res[0][0]) / (a) * 100
        
        col3.subheader("Profit %")
        fo = "{:.2f}".format(return1)
        col3.write(fo)

        st.header("Visualisation:")
        idx = pd.date_range("2021-04-01", periods=len(f_predict), freq="D")
        ts = pd.Series(range(len(idx)), index=idx)
        r = scaler.inverse_transform([f_predict]).reshape(-1, 1)
        r = r.reshape(r.shape[0])
        fig, ax = plt.subplots()
        ax = sns.lineplot(x=df.DateAndTime, y=df['open'], color='r')
        ax = sns.lineplot(x=idx, y=r)
        plt.xticks(rotation = 90)
        st.pyplot(fig)

    # MSFt
    if option == "MSFT 1- Day":
        'You selected:', option

        df = pickle.load(open("df_msft.pkl", "rb"))
        scaler = pickle.load(open('scalerMSFT.pkl', 'rb'))
        model = load_model('modelmsft.h5')
       

        with open('ftestMSFT.pkl', 'rb') as f:
            f_test = pickle.load(f)

        f_test = np.array(f_test)
        f_test = np.reshape(f_test, (f_test.shape[0], f_test.shape[1], 1))

        
        # day, hour, minute = user_input_features()
        day = user_input_features2()

        f_predict = []
        n_days = day

        for i in range(n_days):
            res = model.predict(f_test)
            f_predict.append(res[0][0])
            f_test = np.delete(f_test, [0], None)
            f_test = np.append(f_test, res[0][0], None)
            f_test = f_test.reshape(1, 60, 1)

        st.header("Prediction:")
        res = scaler.inverse_transform([[f_predict[day - 1]]])
        col1, col2, col3 = st.beta_columns(3)

       
        col1.subheader("Open Price")
        col1.write(res[0][0])

       
        col2.subheader("Prev Day Open")
        a = scaler.inverse_transform([[f_predict[day - 2]]])[0][0]
        col2.write(a)

        return1 = (a - res[0][0]) / (a) * 100
        
        fo2 = "{:.2f}".format(return1)
        col3.subheader("Profit %")
        col3.write(fo2)

        st.header("Visualisation:")
        idx = pd.date_range("2021-05-04", periods=len(f_predict), freq="D")
        ts = pd.Series(range(len(idx)), index=idx)
        r = scaler.inverse_transform([f_predict]).reshape(-1, 1)
        r = r.reshape(r.shape[0])
        fig8, ax1 = plt.subplots()

        ax1 = sns.lineplot(x=df.Date[df.Date.dt.year > 2019], y=df['Open'], color='r')
        ax1 = sns.lineplot(x=idx, y=r)
        plt.xticks(rotation = 90)

        st.pyplot(fig8)
st.markdown(

            "<center><h3 style='text-align: centre; color: red;'>Developed By: AASHISH BHARDWAJ,KUNAL SHARMA,VRUNDA,PRITI</h3></center>",
            unsafe_allow_html=True)
