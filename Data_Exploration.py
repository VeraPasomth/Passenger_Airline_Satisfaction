#Data Exploration

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Load Preprocessed Data
df = pd.read_csv("Preprocessed_Airline_Passenger_Satisfaction.csv", index_col=False)
QuantitativeDF = pd.DataFrame(df, columns=['Age','Flight_Distance','Arrival_Delay'])
df = df.drop('Unnamed: 0', axis=1)


Q1 = df['Arrival_Delay'].quantile(0.25)
Q3 = df['Arrival_Delay'].quantile(0.75)
IQR = Q3 - Q1
res = Q3 + (1.5 * IQR)
print(res)


#Display Removal of outliers
fig, axes = plt.subplots(1, 2)
sns.scatterplot(x="Arrival_Delay", y = np.random.rand(df.shape[0]), data=df,hue="Arrival_Delay",ax=axes[0])


df = df[df['Arrival_Delay'] <=500]

sns.scatterplot(x="Arrival_Delay", y = np.random.rand(df.shape[0]), data=df, hue="Arrival_Delay",ax=axes[1],)
sns.set(rc={'figure.figsize':(30,40)})
plt.show()

#Display Flight_Distance and Age Distribution
figure, axes = plt.subplots(1, 2)
sns.histplot(df,x='Flight_Distance',ax=axes[0])
sns.histplot(df,x='Age',ax=axes[1])
plt.show()

#Descriptive Statistics of Quantitative Variables
print('Descriptive Statistics')
print(QuantitativeDF.describe())
print(QuantitativeDF.describe().to_csv('Descriptive Statistics.csv'))

#Check if dataset is balanced
count = df['Satisfaction'].value_counts()
print(count)
ax = df['Satisfaction'].value_counts(normalize=False).plot(kind='bar',
                                                          color=['indianred','steelblue'],
                                                          alpha=0.9, rot=0,
                                                          figsize=(7, 5.8),
                                                          title="Neutral or Unsatisfied VS Satisfied")
ax.set_xlabel("Satisfied = 1 and Neutral or Unsatisfied = 0")
plt.show()


#Display Correlations
def correlation_matrice(corr,dataFrame):
    corr = dataFrame.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool8))
    f, ax = plt.subplots(figsize=(20, 9))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    annot_kws = {'fontsize': 6}
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None,square=True,
                annot=True,annot_kws=annot_kws ,linewidths=.5)
    plt.show()


#Heatmap Correlation
corr3 = df.corr()
correlation_matrice(corr3,df)

#Update Preprocessed CSV File
df = df.drop('Departure_Delay', axis=1)
df.to_csv('Preprocessed_Airline_Passenger_Satisfaction.csv')