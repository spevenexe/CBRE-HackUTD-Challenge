from textblob import TextBlob
import pandas as pd

#use textblob to get subjectivity... 
def getSubjectivity(text):
    return TextBlob(str(text)).sentiment.subjectivity
#..and polarity
def getPolarity(text):
    return TextBlob(str(text)).sentiment.polarity

reviews_data = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
reviews_data.rename(columns={'Unnamed: 0':'Review ID'},inplace=True)

#add new sentimentality columns
reviews_data['Subjectivity'] = reviews_data['Review Text'].apply(getSubjectivity)
reviews_data['Polarity'] = reviews_data['Review Text'].apply(getPolarity)

#sort reviews_data, then write to external file
sorted_data = reviews_data.sort_values(by=['Polarity'])
sorted_data.to_csv('analyzed_data.csv',index=False)