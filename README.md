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

MODEL PREPARED ON LSTM & GRU
For 1. "DataFrame.csv",-----> LSTM
Root Mean Square Error for Trained Data is : 0.0056669180987400265
Root Mean Square Error for Test Data is : 0.006346859081094111

R2 Score for Trained Model is : 0.9994406225111279
R2 Score for Test Model is : 0.9973499198188315

For 1. "DataFrame.csv",-----> GRU
Root Mean Square Error for Trained Data is : 0.013171538050559947
Root Mean Square Error for Test Data is : 0.015511692524527944

R2 Score for Trained Model is : 0.9969780721261131
R2 Score for Test Model is : 0.9841707740783997

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

MODEL PREPARED ON LSTM & GRU
For 1. "MSFT.csv",-----> LSTM
Root Mean Square Error for Trained Data is : 0.002004906519085069
Root Mean Square Error for Test Data is : 0.040218558743184314

R2 Score for Trained Model is : 0.9986558486327152
R2 Score for Test Model is : 0.9664974018277981

For 1. "DataFrame.csv",-----> GRU
Root Mean Square Error for Trained Data is : 0.0037504085130987885
Root Mean Square Error for Test Data is : 0.014248275173150728

R2 Score for Trained Model is : 0.9952965441280276
R2 Score for Test Model is :0.9957951542413979
***************************************************************************************************

<<<<<<< HEAD

=======
>>>>>>> 112da0d00c1bbbf097fb4f600a031ee6953b8d8c

#### Demo:


#### Report File:

[Prediction-of-Stock-Price-Movement-based-on-trading-DS-II.docx](https://github.com/Technocolabs100/Prediction-of-Stock-Price-Movement-based-on-trading-DS-II/files/6803673/Prediction-of-Stock-Price-Movement-based-on-trading-DS-II.docx)



