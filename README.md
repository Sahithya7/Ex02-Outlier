# Ex02-Outlier

You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
# EXPLANATION
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.
# ALGORITHM
# STEP 1
Read the given Data
# STEP 2
Get the information about the data
# STEP 3
Detect the Outliers using IQR method and Z score
# STEP 4
Remove the outliers
# STEP 5
Plot the datas using Box Plot
# CODE
(1) & (2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe
```
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("C:\Users\chief\OneDrive\Documents\Ex02-Outlier\bhp.csv")
df

df.head()

df.describe()

df.info()

df.isnull().sum()

df.shape

sns.boxplot(x="price_per_sqft",data=df)
q1 = df['price_per_sqft'].quantile(0.25)
q3 = df['price_Aper_sqft'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df1 =df[((df['price_per_sqft']>=ll)&(df['price_per_sqft']<=ul))]
df1

df1.shape

sns.boxplot(x="price_per_sqft",data=df1)
(3) Examine price_per_sqft column and use zscore of 3 to remove outliers.
from scipy import stats

z = np.abs(stats.zscore(df['price_per_sqft']))
df2 = df[(z<3)]
df2

print(df2.shape)
sns.boxplot(x="price_per_sqft",data=df2)
(4)(i) For the data set height_weight.csv detect weight outliers using IQR method
df3 = pd.read_csv("C:\Users\chief\OneDrive\Documents\Ex02-Outlier\height_weight.csv")
df3

df3.head()

df3.info()

df3.describe()

df3.isnull().sum()

df3.shape
sns.boxplot(x="weight",data=df3)

q1 = df3['weight'].quantile(0.25)
q3 = df3['weight'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df4 =df3[((df3['weight']>=ll)&(df3['weight']<=ul))]
df4

df4.shape

sns.boxplot(x="weight",data=df4)(1)(2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe
# Dataset

(4)(ii) For the data set height_weight.csv detect height outliers using IQR method
sns.boxplot(x="height",data=df3)

q1 = df3['height'].quantile(0.25)
q3 = df3['height'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df5 =df3[((df3['height']>=ll)&(df3['height']<=ul))]
df5

df5.shape

sns.boxplot(x="height",data=df5)
```
# OUTPUT
(1)(2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe
# Dataset
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/b79fa48d-11d3-4d2c-baec-35809d12a40f)
# Dataset Head
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/134b9736-94e7-4e52-a37e-e4ed0eb14fbd)
# Dataset Info
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/0ee560a8-6d9b-460e-8cae-3cb3c9fff425)
# Dataset Describe
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/a1607c1e-7fc6-461a-bae9-918df5c501b4)
# Null Values!
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/6fa570f7-b9c5-46d0-9e6a-6f27388c1598)
# Dataset Shape
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/340ea93d-6874-4424-b5a4-6be3b1e2fc48)
# Box plot of price_per_sqft column with outliers
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/e877ab2a-2873-46f7-aafa-42dee96696de)
# price_per_sqft - Dataset after removing outliers
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/73b2912d-2dc8-425e-b8d2-501fe94f9812)
# price_per_sqft - Shape of Dataset after removing outliers
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/b889c076-1046-4241-a3d1-87069c334a4f)
# Box Plot of price_per_sqft column without outliers
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/e10c8d92-78db-4226-93f8-da638b56a839)
# (3) Examine price_per_sqft column and use zscore of 3 to remove outliers.
# Dataset after removal of outlier using z score
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/3ea833ca-0014-41b1-91a6-dbd1b69ff863)
# Shape of Dataset after removal of outlier using z score
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/3ea88baf-5ed6-4dee-86b8-2b8b8a635449)
# price_per_sqft column after removing outliers
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/c0b01681-f934-43dd-a396-a15a52051f4c)
# (4) For the data set height_weight.csv detect weight and height outliers using IQR method
# Dataset
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/af4a3392-8998-47c2-baa9-2fee92a38a23)
# Dataset Head
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/04e90265-b57b-4d6c-8943-94b6badd4ba3)
# Dataset Info
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/a6d111b3-a235-4254-9d4e-2b2709c009af)
# Dataset Describe
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/914d57bc-7481-4dc2-b7e3-c4c48e6e997a)
# Null Values
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/918d23bf-3c79-43f3-84cf-aeb29c9f3a52)
# Dataset Shape
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/d0f8b660-161a-4e6c-a140-aae83f77b749)
# Weight - With outliers
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/264b942b-cd62-4abb-97d4-c6cd6d729803)
# Weight - Dataset after removing Outliers using IQR method
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/3f689ac4-c639-4ea3-9fd4-f095c5486ee1)
# Weight - Shape of Dataset after removing Outliers using IQR method
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/8be6a321-7861-44a6-b983-4af8af37aa9a)
# Weight - Without Outliers using IQR method
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/1dc43f4d-9bb2-461e-b6db-080380dc13ca)
# Height - With outliers
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/8430b9fc-132c-4187-bb74-4bc97a03f2ba)
# Height - Dataset after removing Outliers using IQR method
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/d58722d3-a2c1-4173-b166-6773a447b0f3)
# Height - Shape of Dataset after removing Outliers using IQR method
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/9308458d-31bc-4bf8-971c-d6ca225a2914)
# Height - Without Outliers using IQR method
![image](https://github.com/Sahithya7/Ex02-Outlier/assets/133002193/94cf2018-aef1-44aa-aaae-0c4178721be9)
# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods. And print them
