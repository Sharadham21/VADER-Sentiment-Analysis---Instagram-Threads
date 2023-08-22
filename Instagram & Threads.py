import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib as fm
import seaborn as sns

plt.style.use('ggplot')

import nltk


#Instagram Playstore Reviews - Sentiment Analysis Project

#Loading Dataset
Instagram_Playstore_Reviews_df = pd.read_csv("C:\\Users\\shara\\OneDrive\\Desktop\\Projects - Instagram Sentiment Analysis\\Instagram Playstore Reviews - Kaggle Dataset.csv", encoding='unicode_escape')
#Renaming Columns
Instagram_Playstore_Reviews_df.rename(columns = {'review_description':'Review'}, inplace = True)
Instagram_Playstore_Reviews_df.rename(columns = {'rating':'Rating (0 - 5)'}, inplace = True)
Instagram_Playstore_Reviews_df.rename(columns = {'review_date':'Date'}, inplace = True)
#Removing rows with null values
Instagram_Playstore_Reviews_df = Instagram_Playstore_Reviews_df.dropna()
#Converting the column containing timestamps to datetime format
Instagram_Playstore_Reviews_df['Date'] = pd.to_datetime(Instagram_Playstore_Reviews_df['Date'])
#Extracting only the Date part from the Datetime
Instagram_Playstore_Reviews_df['Date_New'] = Instagram_Playstore_Reviews_df['Date'].dt.date
#Dropping the original timestamp column
Instagram_Playstore_Reviews_df.drop('Date', axis=1, inplace=True)
#Renaming "Date_New" to Date
Instagram_Playstore_Reviews_df.rename(columns = {'Date_New':'Date'}, inplace = True)
#Moving the Date column to the first index position
Date_Column = Instagram_Playstore_Reviews_df.pop('Date')
Instagram_Playstore_Reviews_df.insert(0, 'Date', Date_Column)
#Sorting the dataframe in Chronological Order (Ascending)
Instagram_Playstore_Reviews_df = Instagram_Playstore_Reviews_df.sort_index()
#Filtering rows for the year 2023
Instagram_Playstore_Reviews_df['Date'] = pd.to_datetime(Instagram_Playstore_Reviews_df['Date'])
Instagram_Playstore_2023 = Instagram_Playstore_Reviews_df[(Instagram_Playstore_Reviews_df['Date'].dt.year == 2023)]
#Extract only the date part (without the timestamp) and overwrite the column
Instagram_Playstore_2023['Date'] = Instagram_Playstore_2023['Date'].dt.date
Instagram_Playstore_2023.reset_index(drop=True, inplace=True)
Instagram_Playstore_2023 = Instagram_Playstore_2023.rename_axis('Index_No')
#Extracting the first 500 rows of each Dataframe
#2023
Instagram_Playstore_2023 = Instagram_Playstore_2023.head(500)

#Plotting the Rating (0-5) Count from the 2023 sample
plt.rcParams["font.family"] = "serif"  # Set the font family to a serif font
color_palette = ['#a0b3a4', '#61766e', '#7b0a73', '#e7af4f', '#e40c08']
ax2023_bar1 = Instagram_Playstore_2023['Rating (0 - 5)'].value_counts().sort_index() \
        .plot(kind='bar' ,
              title='Count of Instagram Ratings by Rating Scores - 2023' ,
              figsize=(10,5),
              color=color_palette)  # Specify the desired color
ax2023_bar1.set_xlabel('Rating (0-5)', fontdict={'color': 'black'})
ax2023_bar1.set_ylabel('Count', fontdict={'color': 'black'})
tick_label_style = {'color': 'black', 'size': 'medium', 'weight': 'bold'}
ax2023_bar1.set_xticklabels(ax2023_bar1.get_xticklabels(), fontdict=tick_label_style)
ax2023_bar1.set_yticklabels(ax2023_bar1.get_yticklabels(), fontdict=tick_label_style)
plt.tight_layout()
plt.show()

#VADER Sentiment Scoring - Instagram
# We will use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text.
# This uses a "Bag of words" approach:
 #1. Stop words are removed
 #2. Each word is scored and combined to a total score

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia = SentimentIntensityAnalyzer()
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Create a list to store the results
results = []
#Iterate through the rows and analyze sentiment
for index, row in Instagram_Playstore_2023.iterrows():
    text = row['Review']
    sentiment_scores = sia.polarity_scores(text)
    results.append({
        'Review': text,
        'Positive': sentiment_scores['pos'],
        'Negative': sentiment_scores['neg'],
        'Neutral': sentiment_scores['neu'],
        'Compound': sentiment_scores['compound']
    })
#Create a new DataFrame from the results
vadersinstagram2023 = pd.DataFrame(results)
#Print the resulting sentiment DataFrame
print(vadersinstagram2023)
#Merge the DataFrames based on the "Review" column
vadersinstagram2023 = pd.merge(vadersinstagram2023, Instagram_Playstore_2023[['Review', 'Rating (0 - 5)']], on='Review', how='left')


#Plot VADER results - Negative, Positive and Neutral
#Bar Plot
color_palette = ['#a0b3a4', '#61766e', '#7b0a73', '#e7af4f', '#e40c08']
fig, axs = plt.subplots(1, 3, figsize=(15,3))
sns.barplot(data=vadersinstagram2023, x='Rating (0 - 5)', y='Positive', ax=axs[0], palette=color_palette)
sns.barplot(data=vadersinstagram2023, x='Rating (0 - 5)', y='Neutral', ax=axs[1], palette=color_palette)
sns.barplot(data=vadersinstagram2023, x='Rating (0 - 5)', y='Negative', ax=axs[2], palette=color_palette)
axs[0].set_title('Positive - Instagram')
axs[1].set_title('Neutral - Instagram')
axs[2].set_title('Negative - Instagram')
for ax in axs:
    ax.set_xlabel('Rating (0-5)', fontdict={'color': 'black'})
    ax.set_ylabel('Count', fontdict={'color': 'black'})
tick_label_style = {'color': 'black', 'weight': 'bold'}
for ax in axs:
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=tick_label_style)
    ax.set_yticklabels(ax.get_yticklabels(), fontdict=tick_label_style)
plt.tight_layout()
plt.show()

#Pie Chart
#Calculate the counts of each sentiment category
positive_count = vadersinstagram2023['Positive'].sum()
neutral_count = vadersinstagram2023['Neutral'].sum()
negative_count = vadersinstagram2023['Negative'].sum()
#Data for the Pie Chart
sentiment_counts = [positive_count, neutral_count, negative_count]
labels = ['Positive', 'Neutral', 'Negative']
colors = ['#a0b3a4', '#7b0a73', '#e7af4f']
#Create the Pie Chart
plt.figure(figsize=(8, 6))
plt.pie(sentiment_counts, labels=None, colors=colors, autopct=lambda p: '{:.1f}%'.format(p), startangle=140,
        textprops={'fontsize': 14})
plt.title('Proportion of Sentiment Categories for Instagram - 2023', fontdict={'color': 'black', 'fontsize': 20})
plt.legend(labels, loc='lower right', bbox_to_anchor=(1.25,0), prop={'size': 13})
plt.tight_layout()
plt.show()


#Threads Playstore Reviews - Sentiment Analysis Project

#Loading Dataset
Threads_Playstore_Reviews_2023_df = pd.read_csv("C:\\Users\\shara\\OneDrive\\Desktop\\Projects - Instagram Sentiment Analysis\\Threads Playstore Reviews - Kaggle Dataset.csv", encoding='unicode_escape')
#Dropping the "source" column
Threads_Playstore_Reviews_2023_df.drop(columns=['source'], inplace=True)
#Renaming Columns
Threads_Playstore_Reviews_2023_df.rename(columns = {'review_description':'Review'}, inplace = True)
Threads_Playstore_Reviews_2023_df.rename(columns = {'rating':'Rating (0 - 5)'}, inplace = True)
Threads_Playstore_Reviews_2023_df.rename(columns = {'review_date':'Date'}, inplace = True)
#Removing rows with null values
Threads_Playstore_Reviews_2023_df = Threads_Playstore_Reviews_2023_df.dropna()
#Converting the column containing timestamps to datetime format
Threads_Playstore_Reviews_2023_df['Date'] = pd.to_datetime(Threads_Playstore_Reviews_2023_df['Date'])
#Extracting only the Date part from the Datetime
Threads_Playstore_Reviews_2023_df['Date_New'] = Threads_Playstore_Reviews_2023_df['Date'].dt.date
#Dropping the original timestamp column
Threads_Playstore_Reviews_2023_df.drop('Date', axis=1, inplace=True)
#Renaming "Date_New" to Date
Threads_Playstore_Reviews_2023_df.rename(columns = {'Date_New':'Date'}, inplace = True)
#Moving the Date column to the first index position
Date_Column = Threads_Playstore_Reviews_2023_df.pop('Date')
Threads_Playstore_Reviews_2023_df.insert(0, 'Date', Date_Column)
Threads_Playstore_Reviews_2023_df.reset_index(drop=True, inplace=True)
Threads_Playstore_Reviews_2023_df = Threads_Playstore_Reviews_2023_df.rename_axis('Index_No')
#Extracting the first 500 rows of each Dataframe
#2023
Threads_Playstore_Reviews_2023_df = Threads_Playstore_Reviews_2023_df.head(500)

#Plotting the Rating (0-5) Count
plt.rcParams["font.family"] = "serif"  # Set the font family to a serif font
color_palette = ['#300c64', '#6449a6', '#ce4969', '#e58835', '#eec91c']
ax2023_bar1 = Threads_Playstore_Reviews_2023_df['Rating (0 - 5)'].value_counts().sort_index() \
        .plot(kind='bar' ,
              title='Count of Ratings for Thread by Rating Scores - 2023' ,
              figsize=(10,5),
              color=color_palette)  # Specify the desired color
ax2023_bar1.set_xlabel('Rating (0-5)', fontdict={'color': 'black'})
ax2023_bar1.set_ylabel('Count', fontdict={'color': 'black'})
tick_label_style = {'color': 'black', 'size': 'medium', 'weight': 'bold'}
ax2023_bar1.set_xticklabels(ax2023_bar1.get_xticklabels(), fontdict=tick_label_style)
ax2023_bar1.set_yticklabels(ax2023_bar1.get_yticklabels(), fontdict=tick_label_style)
plt.tight_layout()
plt.show()

#VADER Sentiment Scoring - Instagram
# We will use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text.
# This uses a "Bag of words" approach:
 #1. Stop words are removed
 #2. Each word is scored and combined to a total score

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia = SentimentIntensityAnalyzer()
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Create a list to store the results
results = []
#Iterate through the rows and analyze sentiment
for index, row in Threads_Playstore_Reviews_2023_df.iterrows():
    text = row['Review']
    sentiment_scores = sia.polarity_scores(text)
    results.append({
        'Review': text,
        'Positive': sentiment_scores['pos'],
        'Negative': sentiment_scores['neg'],
        'Neutral': sentiment_scores['neu'],
        'Compound': sentiment_scores['compound']
    })
#Create a new DataFrame from the results
vadersthreads2023 = pd.DataFrame(results)
#Print the resulting sentiment DataFrame
print(vadersthreads2023)
#Merge the DataFrames based on the "Review" column
vadersthreads2023 = pd.merge(vadersthreads2023, Threads_Playstore_Reviews_2023_df[['Review', 'Rating (0 - 5)']], on='Review', how='left')

#Plot VADER results - Negative, Positive and Neutral
#Bar Plot
color_palette = ['#300c64', '#6449a6', '#ce4969', '#e58835', '#eec91c']
fig, axs = plt.subplots(1, 3, figsize=(15,3))
sns.barplot(data=vadersthreads2023, x='Rating (0 - 5)', y='Positive', ax=axs[0], palette=color_palette)
sns.barplot(data=vadersthreads2023, x='Rating (0 - 5)', y='Neutral', ax=axs[1], palette=color_palette)
sns.barplot(data=vadersthreads2023, x='Rating (0 - 5)', y='Negative', ax=axs[2], palette=color_palette)
axs[0].set_title('Positive - Threads')
axs[1].set_title('Neutral - Threads')
axs[2].set_title('Negative - Threads')
for ax in axs:
    ax.set_xlabel('Rating (0-5)', fontdict={'color': 'black'})
    ax.set_ylabel('Count', fontdict={'color': 'black'})
tick_label_style = {'color': 'black', 'weight': 'bold'}
for ax in axs:
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=tick_label_style)
    ax.set_yticklabels(ax.get_yticklabels(), fontdict=tick_label_style)
plt.tight_layout()
plt.show()

#Pie Chart
#Calculate the counts of each sentiment category
positive_count = vadersthreads2023['Positive'].sum()
neutral_count = vadersthreads2023['Neutral'].sum()
negative_count = vadersthreads2023['Negative'].sum()
#Data for the Pie Chart
sentiment_counts = [positive_count, neutral_count, negative_count]
labels = ['Positive', 'Neutral', 'Negative']
colors = ['#6449a6', '#ce4969', '#eec91c']
#Create the Pie Chart
plt.figure(figsize=(8, 6))
plt.pie(sentiment_counts, labels=None, colors=colors, autopct=lambda p: '{:.1f}%'.format(p), startangle=140,
        textprops={'fontsize': 14})
plt.title('Proportion of Sentiment Categories for Threads - 2023', fontdict={'color': 'black', 'fontsize': 20})
plt.legend(labels, loc='lower right', bbox_to_anchor=(1.25,0), prop={'size': 13})
plt.tight_layout()
plt.show()





