import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib as fm
import seaborn as sns
plt.style.use('ggplot')

#Loading Dataset
vadersinstagram2023_df = pd.read_csv("C:\\Users\\shara\\OneDrive\\Desktop\\Projects - Instagram Sentiment Analysis\\vadersinstagram2023.csv", encoding='unicode_escape')
vadersthreads2023_df = pd.read_csv("C:\\Users\\shara\\OneDrive\\Desktop\\Projects - Instagram Sentiment Analysis\\vadersthreads2023.csv", encoding='unicode_escape')
#Dropping the "Unnamed" column
vadersinstagram2023_df.drop(columns=['Unnamed: 0'], inplace=True)
vadersthreads2023_df.drop(columns=['Unnamed: 0'], inplace=True)
#Renaming Columns - Instagram
vadersinstagram2023_df.rename(columns = {'Review':'Review - Instagram'}, inplace = True)
vadersinstagram2023_df.rename(columns = {'Positive':'Positive - Instagram'}, inplace = True)
vadersinstagram2023_df.rename(columns = {'Negative':'Negative - Instagram'}, inplace = True)
vadersinstagram2023_df.rename(columns = {'Neutral':'Neutral - Instagram'}, inplace = True)
vadersinstagram2023_df.rename(columns = {'Compound':'Compound - Instagram'}, inplace = True)
vadersinstagram2023_df.rename(columns = {'Rating (0 - 5)':'Rating - Instagram'}, inplace = True)
#Renaming Columns - Instagram
vadersthreads2023_df.rename(columns = {'Review':'Review - Threads'}, inplace = True)
vadersthreads2023_df.rename(columns = {'Positive':'Positive - Threads'}, inplace = True)
vadersthreads2023_df.rename(columns = {'Negative':'Negative - Threads'}, inplace = True)
vadersthreads2023_df.rename(columns = {'Neutral':'Neutral - Threads'}, inplace = True)
vadersthreads2023_df.rename(columns = {'Compound':'Compound - Threads'}, inplace = True)
vadersthreads2023_df.rename(columns = {'Rating (0 - 5)':'Rating - Threads'}, inplace = True)

#Pairplot - Comparing VADER Sentiment Scores of Instagram & Threads
ax_pairplot = sns.pairplot(data=vadersinstagram2023_df,
             vars=['Positive - Instagram', 'Negative - Instagram', 'Neutral - Instagram'],
             hue='Rating - Instagram',
             palette='tab10')
plt.show()

ax_pairplot = sns.pairplot(data=vadersthreads2023_df,
             vars=['Positive - Threads', 'Negative - Threads', 'Neutral - Threads'],
             hue='Rating - Threads',
             palette='tab10')
plt.show()








