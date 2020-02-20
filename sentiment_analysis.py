import pandas as pd            #Pandas
from textblob import TextBlob #For Sentiment Anlysis
from itertools import islice 
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns
from IPython.display import display

import nltk

from numpy.random import randn
from numpy.random import seed
from numpy import cov
from scipy.stats import pearsonr

from pylab import rcParams

from pylab import rcParams

from collections import Counter

import matplotlib as mpl

import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import random



if __name__ == "__main__":


    # Use to do sentiment analysis  and save the results as .xlsx file(If done comment it because it's resource consuming)
    '''
    df_data = pd.read_excel("temp.xlsx")
    print(df_data.head())
    print("\n")

    COLS = ['date','text', 'sentiment','subjectivity','polarity']
    df = pd.DataFrame(columns=COLS)
    print(df)
    print("\n")
    #print(df_data.iterrows())
    #print("\n")



    for index, row in islice(df_data.iterrows(), 0, None):

        new_entry = []
        text_lower= (row['text'])
        blob = TextBlob(text_lower)
        sentiment = blob.sentiment

        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity
             
        new_entry += [row['tweet_created'],text_lower,sentiment,subjectivity,polarity]
            
        single_survey_sentimet_df = pd.DataFrame([new_entry], columns=COLS)
        df = df.append(single_survey_sentimet_df, ignore_index=True)

    df.to_excel('sentiment_analysis.xlsx')
    print(df.head())
    print(df.describe())
    '''



    df = pd.read_excel("sentiment_analysis.xlsx")
    print(df.head())
    print("\n")
    print(df.describe())
    print("\n")



    df_filter = df.loc[(df.loc[:, df.dtypes != object] != 0).any(1)]
    print(df_filter.describe())
    print("\n")



    #boxplot for df_filter
    boxplot = df_filter.boxplot(column=['subjectivity','polarity'], 
                         fontsize = 15,grid = True, vert=True,figsize=(10,10,))
    plt.ylabel('Range')
    plt.show()



    #scatter for df_filter
    sns.lmplot(x='subjectivity',y='polarity',data=df_filter,fit_reg=True,scatter=True, height=10,palette="mute")
    plt.show()



    #covariance and correlation for df_filter
    # calculate the covariance between two variables
    # prepare data
    data_1 = df_filter['subjectivity']
    data_2 = data_1 + df_filter['polarity']
    # calculate covariance matrix
    covariance = cov(data_1, data_2)
    # calculate correlation matrix
    correlation = pearsonr(data_1,data_2) 
    print(covariance)
    print("\n")
    print(correlation)
    print("\n")



    #Polarity Distribution for df_filter
    plt.hist(df_filter['polarity'], color = 'darkred', edgecolor = 'black', density=False,
             bins = int(30))
    plt.title('Polarity Distribution')
    plt.xlabel("Polarity")
    plt.ylabel("Number of Times")

    rcParams['figure.figsize'] = 10,15
    plt.show()



    #Polarity Distribution Density for df_filter
    sns.distplot(df_filter['polarity'], hist=True, kde=True, 
                 bins=int(30), color = 'darkred',
                 hist_kws={'edgecolor':'black'},axlabel ='Polarity')
    plt.title('Polarity Density')

    rcParams['figure.figsize'] = 10,15
    plt.show()



    # Frequent word finding(NLP)
    #nltk.download() # Don't use if you already downloaded
    stopwords = nltk.corpus.stopwords.words('english')

    RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))
    words = (df.text
               .str.lower()
               .replace([r'\|',r'\&',r'\-',r'\.',r'\,',r'\'', RE_stopwords], [' ', '','','','','',''], regex=True)
               .str.cat(sep=' ')
               .split()
    )



    # generate DF out of Counter
    rslt = pd.DataFrame(Counter(words).most_common(10),
                        columns=['Word', 'Frequency']).set_index('Word')
    print(rslt)
    print("\n")



    # Bar and PI cahrt ploting
    rslt_wordcloud = pd.DataFrame(Counter(words).most_common(100),
                        columns=['Word', 'Frequency'])
    #BAR CHART
    rslt.plot.bar(rot=40, figsize=(16,10), width=0.8,colormap='tab10')
    plt.title("Commanly used words by Clients (SurveyData - Q7 )")
    plt.ylabel("Count")
    plt.show()

    from pylab import rcParams
    rcParams['figure.figsize'] = 10,15

    #PIE CHART

    explode = (0.1, 0.12, 0.122, 0,0,0,0,0,0,0)  # explode 1st slice
    labels=['@united',
            'flight',
            '@usairways',
            '@amaricamair',
            '@southwestair',
            '@jetblue',
            'get',
            'cancelled',
            'thanks',
            'service']

    plt.pie(rslt['Frequency'], explode=explode,labels =labels , autopct='%1.1f%%',
            shadow=False, startangle=90)
    plt.legend( labels, loc='lower left',fontsize='x-small',markerfirst = True)
    plt.tight_layout()
    plt.title(' Commanly used words in tweets')
    plt.show()

    mpl.rcParams['font.size'] = 15.0




    # Wordcloud
    wordcloud = WordCloud(max_font_size=60, max_words=100, width=480, height=380,colormap="brg",
                          background_color="white").generate(' '.join(rslt_wordcloud['Word']))
                          
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.figure(figsize=[10,10])
    plt.show()