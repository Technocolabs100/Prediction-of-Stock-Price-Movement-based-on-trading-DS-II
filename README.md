# Prediction-of-Stock-Price-Movement-based-on-trading-DS-II

********************************* TEAM B FILE****************************************
# Prediction-of-Stock-Price-Movement-based-on-trading-DS-II
The Raw Datasets that have been taken in this model creation were: Dataframe.csv and MSFT.csv
**********************************************************
BLUEPRINT AS PER TEAM -A,( open , high , low , close , profir/loss , max_profit , adj_close , volume)

[BLUE PRINT TEAM-B (Final) (2).pdf](https://github.com/Technocolabs100/Prediction-of-Stock-Price-Movement-based-on-trading-DS-II/files/6803688/BLUE.PRINT.TEAM-B.Final.2.pdf)


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
Different visualization plots of both the datasets:-
![Screenshot from 2021-06-15 20-30-07](https://user-images.githubusercontent.com/80449168/125417833-aa606c31-6ca3-4904-a4fa-b035d4ce043c.png)
![Screenshot from 2021-06-15 20-30-12](https://user-images.githubusercontent.com/80449168/125417844-ee4ab2c6-7286-41fc-92d2-c5a73b365b99.png)
![Screenshot from 2021-06-15 20-29-07](https://user-images.githubusercontent.com/80449168/125417859-5ab1183a-35e0-4037-b484-119493a608e2.png)
![Screenshot from 2021-06-15 20-29-44](https://user-images.githubusercontent.com/80449168/125417875-f716a03e-ee4f-4bb8-8bf3-624688b32935.png)
![Screenshot from 2021-06-15 20-28-10](https://user-images.githubusercontent.com/80449168/125417894-775ee172-6baa-4955-bdb4-6895bbe66996.png)
![Screenshot from 2021-06-15 20-30-01](https://user-images.githubusercontent.com/80449168/125417913-c332d4cd-2651-47e2-ab98-00b8664e7ab5.png)
![Screenshot from 2021-06-15 20-29-53](https://user-images.githubusercontent.com/80449168/125417935-ac590214-027a-43f4-aef4-c36a54b9745e.png)
![Screenshot from 2021-06-15 20-29-18](https://user-images.githubusercontent.com/80449168/125417942-e2c579c8-bf64-4381-807c-509bf3071fe7.png)
![Screenshot from 2021-06-15 20-28-34](https://user-images.githubusercontent.com/80449168/125417959-af21a2a3-0777-4d84-b343-3cceb8b86a83.png)
![Screenshot from 2021-06-15 20-28-23](https://user-images.githubusercontent.com/80449168/125417983-58a63034-d4fe-4ae8-b579-a7c33fad079d.png)

#### Demo


#### Report File
[TECHNOCOLABS DATA SCIENCE.pdf](https://github.com/Technocolabs100/Prediction-of-Stock-Price-Movement-based-on-trading-DS-II/files/6807135/TECHNOCOLABS.DATA.SCIENCE.pdf)


