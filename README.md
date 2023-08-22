# VADER-Sentiment-Analysis---Instagram-Threads
Instagram is a social media platform that has become one of the most popular and widely used worldwide. Instagram has a massive user base of over a billion active monthly users worldwide, highlighting its global presence. 
Threads is another standalone social media platform developed by Facebook, Inc. to create a personalized space for more focused and intimate communication. 
Considering the prevalence of social media in today's world, the sentiment surrounding such platforms is of relevance when studying societal trends and patterns. 

ABOUT DATASETS: 

Instagram Play Store Reviews - Kaggle 
  The dataset obtained from Kaggle entails an extensive collection of user reviews on the Google Play Store for the social media platform Instagram. Kaggle contributor Saloni Jhalani obtained the data by scraping Instagram App reviews from the Google Play Store. The dataset entails user reviews, review date and user ratings on a scale of 1 to 5. 
  https://www.kaggle.com/datasets/saloni1712/instagram-play-store-reviews

Threads Reviews - Kaggle 
  The dataset from Kaggle entails an extensive collection of user reviews on the Google Play Store for the social media platform Threads. Kaggle contributor Saloni Jhalani obtained the data by scraping Instagram App reviews from the Google Play Store and App Store. The dataset entails user reviews, review date and user ratings on a scale of 1 to 5. 
  https://www.kaggle.com/datasets/saloni1712/threads-an-instagram-app-reviews

Sentiment Analysis Method - VADER: 

  VADER (Valence Aware Dictionary and sEntiment Reasoner) Sentiment Analysis is a powerful tool utilized in natural language processing. VADER enables users to determine the overall sentiment in a piece of text, typically short and informal, thus making it highly suitable for analyzing user reviews and comments. The SentimentIntensityAnalyzer which is part of Python's Natural Language Toolkit (nltk) library was utilised for this study. 

ADVANTAGES - VADER SENTIMENT ANALYSIS

  Enhanced Efficiency & Scalability: VADER eliminated the requirement for manual text analysis, saving considerable time and effort. Due to the evergrowing volume of online content, VADER allows for analysing large volumes of data at scale. 

  Informed Decision Making: Regarding the business context, organisations can utilise the VADER tool to identify areas with overall positive sentiment for marketing purposes and areas of negative sentiment for improvement. 

  Trend Identification & Analysis: VADER can be utilised to identify the general sentiment on trending themes and topics, enabling organisations and institutions to remain cognisant of public opinions. 

LIMITATIONS - VADER SENTIMENT ANALYSIS

  Lack of Contextual Understanding: VADER does not entirely grasp the nuances of language since it analysis each word individually. This, in turn, leaves room for misinterpretation, thus compromising the accuracy of the model. 

  Inaccurate Negation Handling: VADER attempts to handle negation phrases such as "not impressed"; however, there is a possibility that it may not always accurately adjust the sentiment in the case of negation. 

  Intensity: VADER does not account for differences in intensity levels and intention behind words and phrases. 

  Emoticons: VADER does have the ability to interpret emoticons. However, it may struggle to accurately capture the sentiment behind intricate combinations of text and emoticons. 

  Cultural and Linguistic Nuances: The VADER tool's dictionary comprises the English Language and its linguistic and cultural nuances. Therefore, its effectiveness is compromised for languages with different cultural nuances and linguistic patterns. 

PYTHON PACKAGES UTILISED
  Pandas
  
  Numpy
  
  Matplotlib
  
  Seaborn
  
  nltk
  
  tqdm
  
  ipywidgets 
