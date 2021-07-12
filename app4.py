#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd    #importing necessary libraries
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.svm import SVR
import streamlit as st

from sklearn import datasets
import yfinance as yf
import urllib
from PIL import Image
#from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf


# In[ ]:





# In[22]:


filename="stock_dataframe_linear.pkl"
fileobj=open(filename,'rb')
svr1=pickle.load(fileobj)


# In[23]:


filename="stock_dataframe_poly.pkl"
fileobj=open(filename,'rb')
svr2=pickle.load(fileobj)


# In[24]:


filename="stock_MSFT_linear.pkl"
fileobj=open(filename,'rb')
svr3=pickle.load(fileobj)


# In[25]:


filename="stock_MSFT_poly.pkl"
fileobj=open(filename,'rb')
svr4=pickle.load(fileobj)


# In[26]:


filename="stock_dataframe_sc.pkl"
fileobj=open(filename,'rb')
sc1=pickle.load(fileobj)


# In[27]:


filename="stock_MSFT_sc.pkl"
fileobj=open(filename,'rb')
sc2=pickle.load(fileobj)


# In[ ]:





# In[28]:


if st.checkbox("Show Credits"):
    st.sidebar.markdown("<h1 style='text-align: left; color: green;'>Welcome!</h1>",
                        unsafe_allow_html=True)

    urllib.request.urlretrieve("https://technocolabs.tech//assets//img//logo1.png", "logo1.png")
    # img = Image.open("logo1.png")
    # img.show()
    img = Image.open("logo1.png")

    # st.text[website](https://technocolabs.tech/)
    # display image using streamlit
    # width is used to set the width of an image
    st.sidebar.image(img, width=200)

    st.sidebar.subheader("Credits")

    st.sidebar.subheader("Under Guidance of")
    # **Guidance @ CDAC-ACTS, Pune**\n
    st.sidebar.info(
        """
        1. Yasin Sir\n
        2. Deepika\n
        3. Team @ [Technocolab](https://www.linkedin.com/company/technocolabs/)\n
        """)
    st.sidebar.subheader("Contributors/Project Team")
    st.sidebar.info(
        "1. [Abhishek Sharma](https://www.linkedin.com/in/abhishek-sharma-285a48205/)\n"
        "2. [Rahul Chawla](https://www.linkedin.com/in/rahul-chawla-942b47150/)\n"
        "3. [Anukriti Singh]( https://www.linkedin.com/mwlite/in/anukriti-singh-3b424a19b/)\n"
        "4. [Yuvraj Kumar](https://www.linkedin.com/in/yuvraj-kumar-68164117a/)\n"
        "5. [Swetha Srinivasan](https://www.linkedin.com/in/swethas-25072001/)\n"
        "6. [Navya Cherian](https://www.linkedin.com/in/navya-cherian/)\n"
        "7. [Gaurav Singh](https://www.linkedin.com/in/gaurav-singh-9b3399187/)"
    )
    st.sidebar.subheader("Project Report")
    st.sidebar.info("[Project Report](https://drive.google.com/file/d/1EaQiElehdcjD4pj4J9quqvn9MZLokmrz/view?usp=sharing)\n")
    st.sidebar.subheader("Connect with Technocollab")
    st.sidebar.info("[contact us](https://technocolabs.tech/)\n")
st.markdown('''
# Stock Price App
Shown are the stock price data for query companies!
**Credits**
- App built by [Technocolab team b](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor))
- Built in `Python` using `streamlit`,`yfinance`, `cufflinks`, `pandas` and `datetime`
''')


# In[29]:


# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

# Retrieving tickers data
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker

# Ticker information
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

# Ticker data
st.header('**Ticker data**')
st.write(tickerDf)

# Bollinger bands
st.header('**Bollinger Bands**')
qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

####
#st.write('---')
#st.write(tickerData.info)


# In[30]:


my_expander3 = st.beta_expander("Visualization with Indicators", expanded=False)
with my_expander3:
    df = pd.read_csv("test_002.csv")

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
        st.write("Sorry! This feature is under construction, the result may vary.")
        coMA1, coMA2 = st.beta_columns(2)

        with coMA1:
            numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)

        with coMA2:
            windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)

        start = dt.datetime.today() - dt.timedelta(numYearMA * 365)
        end = dt.datetime.today()
        # dataMA = yf.download(ticker, start, end)
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
   


# In[31]:


st.subheader('User Input parameters')
def user_input_features():
    OPEN = st.sidebar.slider('OPEN', 13604, 15430, 14703)
    HOUR = st.sidebar.slider('HOUR', 0, 12, 12)
    MINUTE = st.sidebar.slider('MINUTE', 0, 59, 29)
    calendar_ip = st.date_input('DATE')
    DAY = calendar_ip.day
    MONTH = calendar_ip.month
    YEAR = calendar_ip.year
    data = {'OPEN': OPEN,
            'HOUR': HOUR,
            'MINUTE': MINUTE,
            'DAY': DAY,
            'MONTH': MONTH,
            'YEAR': YEAR}
    features = (data)
    return features


# In[32]:


def user_input_features_1():
    OPEN=st.sidebar.slider('OPEN',34.83,261.66,34.83)
    calendar_ip = st.date_input('DATE')
    DAY = calendar_ip.day
    MONTH = calendar_ip.month
    YEAR = calendar_ip.year
    data={'OPEN': OPEN,
         'DAY': DAY,
         'MONTH': MONTH,
         'YEAR': YEAR}
    features=(data)
    return features


# In[33]:


option_1 = st.selectbox( 'Which data do you want  ?',('DATAFRAME', 'MSFT'))


# In[34]:


option = st.selectbox( 'Which SVM model do you want  ?',('linear', 'polynomial'))


# In[35]:


if option_1 == 'DATAFRAME':
    df = pd.DataFrame(data = user_input_features(), index = [0])
else:
    df = pd.DataFrame(data = user_input_features_1(), index = [0])

st.subheader('User input parameters')
st.write(df)


# In[36]:


if option_1 == 'DATAFRAME' and option == 'linear':
    svr=svr1
elif option_1 == 'DATAFRAME' and option == 'polynomial':
    svr=svr2
elif option_1 == 'MSFT' and option == 'linear':
    svr=svr3
else:
    svr=svr4


# In[37]:


if option_1 == 'DATAFRAME':
    sc=sc1
else:
    sc=sc2


# In[38]:


df=sc.transform(df)


# In[39]:


#svr.fit(X_train_std,y_train)

prediction = svr.predict(df)
st.subheader('Prediction')
st.write(prediction)


# In[40]:


################footer - finalized##########
genre = st.radio(
    "Do you Like our project?",
    ('Yes', 'No', 'Not Interested'))
if genre == 'Yes':
    st.write('Thanks! for showing Love.')
    st.write("Connect us through linkendin(link available in credit section)")
elif genre == 'No':
    st.write("Recommend changes! ")
    st.write("Connect us through linkendin(link available in credit section)")
elif genre == 'Not Interested':
    st.write("No worry")
    st.write("Connect us through linkendin(link available in credit section)")

ctn1 = st.beta_container()
ctn1.subheader("**---------------------------------Caution!---------------------------------------**")
ctn1.write("""
This Project is used for only learning and development process. We don't encourage anyone 
to invest in stock based on any data represented here.
""")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




