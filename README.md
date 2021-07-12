# Prediction-of-Stock-Price-Movement-based-on-trading-DS-II

################## TEAM -A #######################

BLUEPRINT AS PER TEAM -A,( open , high , low , close , profir/loss , max_profit , adj_close , volume)

>![image](https://user-images.githubusercontent.com/80449168/122106613-c6470b80-ce37-11eb-8a7c-ef3c670f20e3.png)


DATA CLEANING,
For 1. "Dataframe.csv",
>Deleted "Unnamed:7" Column For "Nan" Values
>Parsed The Date attribute in "datetime64" data type.
>Checked For Duplicate Rows(Not Found)
>Added New Featue to Existing dataset "Profit/Loss" & "Max_Profit"

EDA(Exploratry Data Analysis),
For 1. "DataFrame.csv",
>Distribution PLot is Plotted for each Attribute(Skewness)
> ![image](https://user-images.githubusercontent.com/80449168/122102751-65b5cf80-ce33-11eb-8867-21573c42321a.png)    

>Scatter PLot is Plotted between each Attribute(Trend)
>![image](https://user-images.githubusercontent.com/80449168/122102671-4cad1e80-ce33-11eb-9efc-57fb96b75c9d.png)
    
>Line PLot is PLotted for each attribute with respect to "Date"(Trend w.r.t "Date")
> ![image](https://user-images.githubusercontent.com/80449168/122102817-79f9cc80-ce33-11eb-952b-051352c0577a.png)

>Heat Matrix is Shown For Correlation Between Each Attribute(Linear Relation)
>![image](https://user-images.githubusercontent.com/80449168/122102898-90a02380-ce33-11eb-8106-94a9787e873b.png)

>Violin PLot is displayed for Each Attribute (Outliers)
>![image](https://user-images.githubusercontent.com/80449168/122102957-a57cb700-ce33-11eb-87f4-07ddb2f390ae.png)


*********************************************************************************************

DATA CLEANING,
For 2. "MSFT.csv",
>Checked for "Nan" and Missing Values(NOt found)
>Parsed The Date attribute in "datetime64" data type.
>Checked For Duplicate Rows(Not Found)
>Added New Featue to Existing dataset "Profit/Loss" & "Max_Profit"

EDA(Exploratry Data Analysis),
For 2. "MSFT.csv",
>Distribution PLot is Plotted for each Attribute(Skewness)
>![image](https://user-images.githubusercontent.com/80449168/122103004-b9c0b400-ce33-11eb-8aad-e322f91c4a84.png)

>Scatter PLot is Plotted between each Attribute(Trend)
>![image](https://user-images.githubusercontent.com/80449168/122103076-d4932880-ce33-11eb-8bda-4613de10f707.png)  

>Line PLot is PLotted for each attribute with respect to "Date"(Trend w.r.t "Date")
>![image](https://user-images.githubusercontent.com/80449168/122103148-ea085280-ce33-11eb-934c-38b54adaded7.png)

>Heat Matrix is Shown For Correlation Between Each Attribute(Linear Relation)
>![image](https://user-images.githubusercontent.com/80449168/122103238-060bf400-ce34-11eb-9a6d-aab3de5e5d97.png)

>Box PLot is displayed for Each Attribute (Outliers)
>![image](https://user-images.githubusercontent.com/80449168/122103320-1cb24b00-ce34-11eb-8dbc-7ce82fd01a15.png)


********************************* TEAM B FILE****************************************
# Prediction-of-Stock-Price-Movement-based-on-trading-DS-II
The Raw Datasets that have been taken in this model creation were: Dataframe.csv and MSFT.csv
**********************************************************
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

a. https://drive.google.com/file/d/17hWWCpYWR_LACdEGJVyvqg143FEt-2P9/view?usp=sharing 
b. https://drive.google.com/file/d/1FOM5jAHL0S6SDAPE-0dgwK-TmErLw6ec/view?usp=sharing 
c. https://drive.google.com/file/d/1bmOqztLi9Ngkn2B6RMoLksQc_7-PYJhZ/view?usp=sharing
d. https://drive.google.com/file/d/1iy0rFdobtGX01jJ-8igFarAxHAh5-HYK/view?usp=sharing
e. https://drive.google.com/file/d/1FAZ9AdTPfGS5owWhAOONBN6kaIrcdpB3/view?usp=sharing
f. https://drive.google.com/file/d/1KIpoBPt4jcPJs1cYt6dC8CTLtAhERVYX/view?usp=sharing
g. https://drive.google.com/file/d/1NPOOevN9xmTRkQT92kqu5llhfG8fOM4H/view?usp=sharing
h. https://drive.google.com/file/d/1tQaD4rfsNgleb8HiDXcWacapIaPQ6bG2/view?usp=sharing
i. https://drive.google.com/file/d/1aIG0abaAkU1qFaNmL9RCj9d39vYeWZEY/view?usp=sharing
j. https://drive.google.com/file/d/1UeyWkEBp31aWCjKrdZXmxFDlhpKDYbr2/view?usp=sharing

