# Prediction-of-Stock-Price-Movement-based-on-trading-DS-II

********************************* TEAM B FILE****************************************
# Prediction-of-Stock-Price-Movement-based-on-trading-DS-II
The Raw Datasets that have been taken in this model creation were: Dataframe.csv and MSFT.csv
**********************************************************
BLUEPRINT AS PER TEAM -A,( open , high , low , close , profir/loss , max_profit , adj_close , volume)

>![image](![Uploading BLUE PRINT TEAM-B (Final)_page-0001.jpg…]())


Dataset description:

1. DATAFRAME.csv : 
a. Shape (22805,8)
b. Attributes & data types -> Type (object) , Date (object) , Time (object) , Open (int) , High (int) , Low (int) , Close (int)

2. MSFT.csv :
a. Shape (8857,7)
b. Attributes & data types -> Date (object) , Open (int) , High (int) , Low (int) , Close (int) , Adj close (int) , Volume (int)
**********************************************************
Step1: DATA CLEANING PART:

a. In the Data Cleaning part we had removed the blanks, removed all the null values if any present in both these datasets. 
b. After this, we had then parsed the column named 'Date' and converted it into datetime data type from the object data type.
c. Then we dropped the column with NaN values if any present using the inbuilt drop function available with the pandas library. 
d. Also the duplicate values had been checked in both the dataset using the duplicated function and in our case there were no duplicate values present in both these datasets. 
e. The presence of Outliers in the dataset had been taken care of by calculating the IQR score and limits for Upper and Lower whiskers.
***********************************************************
Step2: EDA ANALYSIS PART:

a. In this we did the Data visualization part and analyses both these datasets with the help of different plots like distribution plot, bar plots, heat matrix, box plots, line charts, scatter plots and violin plots. 
b. The distribution plot has been plotted between various attributes so as to check for the distribution of the data, like whether the data is skewed or nornally distributed. 
c. Also in EDA part we plotted the various bar plots and scatter plots and identifies the data distribution of various attributes with respect to the target attribute. 
d. Also we plotted the Heat Map/Matrix of the both these dataframes and analyzed the correlation betwwen different attributes. 
e. To check for the presence of outliers if any present in these datasets, we plotted the Box plots and found a huge amount of presence of Outliers in the MSFT dataset.
*************************************************************
Links for different visualization plots of both the datasets:-



