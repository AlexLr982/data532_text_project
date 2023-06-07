#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing packages
import requests
import pandas as pd
import re
import nltk
import textacy.preprocessing as tprep
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import plotly
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import ggplot
import wordcloud
response = requests.get(url='https://store.steampowered.com/appreviews/413150?json=1').json()


# In[2]:


#Defining a function for getting reviews when given an app ID
def get_reviews(appid, params={'json':1}):
        url = 'https://store.steampowered.com/appreviews/'
        response = requests.get(url=url+appid, params=params)
        return response.json()


# In[3]:


#Defining a function to get multiple reviews
def get_n_reviews(appid, n=100):
    reviews = []
    cursor = '*'
    params = {
            'json' : 1,
            'filter' : 'all',
            'language' : 'english',
            'day_range' : 9223372036854775807,
            'review_type' : 'all',
            'purchase_type' : 'all'
            }

    while n > 0:
        params['cursor'] = cursor.encode()
        params['num_per_page'] = min(100, n)
        n -= 100

        response = get_reviews(appid, params)
        cursor = response['cursor']
        reviews += response['reviews']

        if len(response['reviews']) < 100: break

    return reviews


# In[4]:


#Getting reviews for Darktide
reviews = get_n_reviews('1361210', 100000)


# In[6]:


#Looking at the reviews to see what needs to be done to make a dataframe
reviews


# In[7]:


df = pd.DataFrame(reviews)[['review', 'voted_up']]
df


# In[8]:


#Making a dataframe with the text data and the rating
df2 = pd.DataFrame(reviews)[['review', 'voted_up','votes_up', 'author']]
df2


# In[9]:


type(reviews)


# In[10]:


#Turning the author data into a frame
df1 = pd.DataFrame(df2['author'].values.tolist())
df1


# In[11]:


#joining the two frames so that information about the author and the review are inthe same dataframe
merged_df = df2.join(df1)
merged_df


# In[ ]:





# In[12]:


#Dropping some columns i didn't need
merged_df['voted_up'] = merged_df['voted_up'].astype(int)
merged_df = merged_df.drop('author', axis = 1)
merged_df = merged_df.drop('steamid', axis = 1)
merged_df


# In[13]:


#Creating a function to clean the data it uses a combination of regexs and textacy preprocessing functions
def normalize(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    text = tprep.remove.brackets(text)
    text = tprep.replace.emojis(text)
    text = re.sub(r'\n\n', '', text)
    text = re.sub(r'\n','', text)
    text = re.sub(r':\s*\'[^\']*\'', '', text)
    text = tprep.remove.punctuation(text)
    text = text.lower()
    return text


# In[14]:


#creating test data for the function
test_text = merged_df.iloc[2,0]
test_text


# In[15]:


#Testing on the data
normalize(test_text)


# In[16]:


#Mapping the new funcion to all the review data
merged_df['review'] = merged_df['review'].map(normalize)


# In[17]:


merged_df


# In[19]:


#Saving the frame to cut down on time
merged_df.to_pickle('review_data')


# In[4]:


review_data = pd.read_pickle('review_data')
review_data


# In[5]:


#Tokenizing the review data using NLTK package
review_data['tokenized'] = review_data['review'].apply(nltk.word_tokenize)


# In[6]:


review_data


# In[7]:


#Getting a distribution of classes
print('Dist of classes:')
print('voted up', str(sum(review_data['voted_up']==1)/len(review_data['voted_up'])*100))
print('Voted down', str(sum(review_data['voted_up']==0)/len(review_data['voted_up'])*100))


# In[8]:


#PLoting playtime at review as a histogram with vote overlay
import plotly.express as px
playtime_hist = px.histogram(review_data, x = 'playtime_at_review', color = 'voted_up')
playtime_hist.show()


# In[9]:


sns.displot(review_data, x = 'playtime_at_review', hue='voted_up', stat = 'probability')


# In[10]:


#Getting means of review playtimes
review_data.groupby('voted_up', as_index = False)['playtime_at_review'].mean()


# In[11]:


#Defining a new column of playtime after review
review_data['playtime_after_review'] = review_data['playtime_forever'] - review_data['playtime_at_review']
review_data


# In[12]:


#getting mean value of playtime after review
review_data.groupby('voted_up', as_index = False)['playtime_after_review'].mean()


# In[13]:


2327.77/4025.73
1-.578


# In[14]:


2536.54/3371.53
1-.7523


# In[15]:



len(review_data.iloc[1,9])


# In[16]:


#Calculating the length of reviews based on tokenized data by applying the len function to the review column
review_data['review_length'] = review_data['tokenized'].apply(len)
review_data


# In[17]:


#Comparing mean review length by positive and negative reviews
review_data.groupby('voted_up', as_index = False)['review_length'].mean()


# In[18]:


#Summary statistics for review length
review_data.groupby('voted_up', as_index = False)['review_length'].describe()


# In[19]:


#Review length histogram
import plotly.express as px
length_dist = px.histogram(review_data, x = 'review_length')
length_dist.show()


# In[20]:


from collections import Counter
counter = Counter()


# In[21]:


counter = review_data['review'].map(counter)
counter


# In[22]:


counter


# In[23]:


import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word,pos='v') for word in words]
    return ' '.join(words)
review_data['lemmatized'] = review_data['review'].apply(lemmatize_words)


# In[24]:


#Removing stop words from the data
#First a stopwords function is defined
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))
def remove_stop(tokens):
    return [t for t in tokens if t.lower() not in stopwords]

tokenize = nltk.word_tokenize

#Then a pipline is created which can apply both stopword removal and tokenization at once
pipeline = [str.lower, tokenize, remove_stop]

def prepare(text, pipeline):
    tokens = text
    for transform in pipeline:
        tokens = transform(tokens)
    return tokens

review_data['token_noStop'] = review_data['review'].apply(prepare, pipeline=pipeline)


# In[25]:


#Making a word list 
words = review_data['token_noStop']
allwords = []
for wordlist in words:
    allwords += wordlist


# In[26]:


#Importing a visualization function
from wordcloud import WordCloud


# In[27]:


#Plotting the top 100 words using word cloud
mostcommon = FreqDist(allwords).most_common(100)
wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(30,10), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Top 100 Most Common Words', fontsize=100)
plt.tight_layout(pad=0)
plt.show()


# In[28]:


#Removing some boring words
def remove_boring(text):
    text = re.sub(r'\bgame\b', '', text)
    text = re.sub(r'\blike\b', '', text)
    text = re.sub(r'\bgames\b', '', text)
    return text
    
    
review_data['review_cleaner'] = review_data['review'].map(remove_boring)


# In[29]:


#Making a second tokenized column with no stop words
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))
def remove_stop(tokens):
    return [t for t in tokens if t.lower() not in stopwords]

tokenize = nltk.word_tokenize

pipeline = [str.lower, tokenize, remove_stop]

def prepare(text, pipeline):
    tokens = text
    for transform in pipeline:
        tokens = transform(tokens)
    return tokens

review_data['token_noStop2'] = review_data['review_cleaner'].apply(prepare, pipeline=pipeline)


# In[30]:


#Cleaner wordlist
words = review_data['token_noStop2']
allwords2 = []
for wordlist in words:
    allwords2 += wordlist


# In[31]:


#Cleaner word cloud
mostcommon2 = FreqDist(allwords2).most_common(100)
wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(str(mostcommon2))
fig = plt.figure(figsize=(30,10), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Top 100 Most Common Words', fontsize=100)
plt.tight_layout(pad=0)
plt.show()


# In[32]:


#splitting into two frames based on user vote
positive = review_data[review_data['voted_up']==1]
negative = review_data[review_data['voted_up']==0]


# In[33]:


negative


# In[34]:


positive


# In[35]:


#Positive word list
positive_words = positive['token_noStop2']
allwords_positive = []
for wordlist in positive_words:
    allwords_positive += wordlist


# In[36]:


#Making a word frequency count list
mostcommon_positive = FreqDist(allwords_positive).most_common(20)
mostcommon_positive


# In[37]:


#word count dataframe
positive_tokensdf = pd.DataFrame(mostcommon_positive, columns = ['Word', 'Count'])
positive_tokensdf


# In[38]:



pos_word = positive_tokensdf.plot(kind='barh', width=0.95)
pos_word.invert_yaxis()


# In[39]:


#Word frequency plot
from matplotlib import pyplot as plt
fig, freq_pos = plt.subplots(figsize =(16, 9))
freq_pos.barh(positive_tokensdf['Word'], positive_tokensdf['Count'])


# In[40]:


#Negative word list
negative_words = negative['token_noStop2']
allwords_negative = []
for wordlist in negative_words:
    allwords_negative += wordlist


# In[41]:


#Negative word list
mostcommon_negative = FreqDist(allwords_negative).most_common(20)
mostcommon_negative


# In[42]:


#Word count dataframe
negative_tokensdf = pd.DataFrame(mostcommon_negative, columns = ['Word', 'Count'])
negative_tokensdf


# In[43]:


#Word frequency plot
from matplotlib import pyplot as plt
fig, freq_neg = plt.subplots(figsize =(16, 9))
freq_neg.barh(negative_tokensdf['Word'], negative_tokensdf['Count'])


# In[44]:


#dataframe for machine learning purposes which only has text data available
ml_df = review_data.drop(columns = ['votes_up', 'num_games_owned', 'num_reviews', 'playtime_forever', 'playtime_last_two_weeks', 'playtime_at_review', 'last_played', 'tokenized'])
ml_df


# In[45]:


#Splitting the data into training and test frames
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(ml_df['review'], ml_df['voted_up'], test_size = 0.2, random_state = 1, stratify = ml_df['voted_up'])


# In[46]:


print('Dist of classes:')
print('voted up', str(sum(Y_train==1)/len(Y_train)*100))
print('Voted down', str(sum(Y_train==0)/len(Y_train)*100))


# In[47]:


#Tf-IDF vectorization 
tfidf = TfidfVectorizer(min_df = 10, ngram_range=(1,1))

X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)


# In[48]:


#Linear support vector machine
from sklearn.svm import LinearSVC
model1 = LinearSVC(random_state = 1, tol=1e-5)
model1.fit(X_train_tf, Y_train)


# In[69]:


#Importing model evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[76]:


#Getting predictions from the support vector machine and then calculating metrics
Y_pred = model1.predict(X_test_tf)
print('accuracy:', accuracy_score(Y_test, Y_pred))
print('precision:', precision_score(Y_test, Y_pred))
print('recall:', recall_score(Y_test, Y_pred))


# In[51]:


#SVM support vector machine
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
confusion_matrix(Y_test, Y_pred)
ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred)


# In[62]:


#Random forest model importation
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()


# In[63]:


#FItting the random forest
forest = forest.fit(X_train_tf, Y_train)


# In[75]:


#Random forest predictions and evaluation metrics
forest_pred = forest.predict(X_test_tf)
print("accuracy:", accuracy_score(Y_test, forest_pred))
print("precision:",precision_score(Y_test, forest_pred))
print('recall:', recall_score(Y_test, forest_pred))


# In[65]:


#Random forest confusion matrix
confusion_matrix(Y_test, forest_pred)
ConfusionMatrixDisplay.from_predictions(Y_test, forest_pred)


# In[66]:


#Gaussian naive bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
Y_pred_NB = gnb.fit(X_train_tf.toarray(), Y_train).predict(X_test_tf.toarray())


# In[78]:


#GNB evaluation metric
print('accuracy:', accuracy_score(Y_test, Y_pred_NB))
print('precision:', precision_score(Y_test, Y_pred_NB))
print('recall:', recall_score(Y_test, Y_pred_NB))


# In[67]:


#GNB confusion matrix
confusion_matrix(Y_test, Y_pred_NB)
ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred_NB)


# In[57]:


param_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt'],
    'max_depth': [3, 6],
    'max_leaf_nodes': [3],
}


# In[58]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

grid_search = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_grid)
grid_search.fit(X_train_tf, Y_train)
print(grid_search.best_estimator_)


# In[61]:


forest2 = RandomForestClassifier(max_depth = 3, max_leaf_nodes= 3, n_estimators = 50)
forest2 = forest.fit(X_train_tf, Y_train, )
forest_pred2 = forest2.predict(X_test_tf)
accuracy_score(Y_test, forest_pred2)


# In[52]:


def remove_boring(text):
    text = re.sub(r'\bgame\b', '', text)
    text = re.sub(r'\blike\b', '', text)
    text = re.sub(r'\bgames\b', '', text)
    return text
    
    
review_data['review_cleaner'] = review_data['review'].map(remove_boring)


# In[53]:


#Counter vectorizing data with stop words removed
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(stop_words = nltk.corpus.stopwords.words('english'), min_df = 5, max_df =.7)

count_vectors = count_vectorizer.fit_transform(review_data['review_cleaner'])


# In[54]:


#Defining an latent Dirichlet Allocation model
from sklearn.decomposition import LatentDirichletAllocation

lda_model = LatentDirichletAllocation(n_components = 10, random_state = 45)
lda_matrix = lda_model.fit_transform(count_vectors)
lda_components = lda_model.components_


# In[55]:


#Defining feature names
feature_names = count_vectorizer.get_feature_names_out()


# In[56]:


#Defining a function for displaying topics
def display_topics(model, features, no_top_words=5):
    for topic, word_vector in enumerate(model.components_):
        total = word_vector.sum()
        largest = word_vector.argsort()[::-1]
        print("\nTopic %02d" % topic)
        for i in range(0, no_top_words):
            print("  %s (%2.2f)" % (features[largest[i]],
                 word_vector[largest[i]]*100/total))


# In[57]:


#Displaying the topics
display_topics(lda_model, feature_names)


# In[58]:


#Using LDAvis to plot results
import pyLDAvis.sklearn
import gensim
lda_display = pyLDAvis.sklearn.prepare(lda_model, count_vectors, count_vectorizer, sort_topics = False)


# In[59]:


pyLDAvis.display(lda_display)


# In[ ]:





# In[ ]:


stopwords = set(nltk.corpus.stopwords.words('english'))


# In[56]:


list(stopwords)


# In[ ]:




