## Concept of Bag of words --> vectorization technique to convert all preprocessed text to Vectors
## we have a dataset - of sms text marked as ham and spam, we will apply a BOW algorith on it from nltk package

import pandas as pd
messages = pd.read_csv("spam.csv", names=['label', 'message'], encoding='latin-1')
print(messages.head(5))

## Data cleaning and Text preprocessing
## stemming, lemmatization, remove stopwords, lower all case, remove special chars
import re
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# let us build a corpus which is our cleaned preprocessed messages
corpus=[]
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])# replace anything other than a-z , A-Z with blank
    review = review.lower()
    review =review.split() # convert it all to list of words
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # take all words without stop words and apply stemming
    review = ' '.join(review) # join all these words to a sentence
    corpus.append(review) # append this review to corpus
# print("\n\n",corpus) # you see huge list of sentences

## let us create bag of words model (to build vector) using sklearn's countVectorizer
## refer documentation to see how to use that method
## choose max_features = choose top 2500 words that have max frequency or words repeated

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500, binary=True) # choose max_features = choose top 2500 words that have max frequency or words repeated
X = cv.fit_transform(corpus).toarray() # apply to corpus to get your vector
# print("\n", X) # you see huge output as huge list of sentences
print(X.shape) #(5572, 2500)

## what we see is a normal BOW we can convert it to binary BOW -> binary=True in above code

## next let see teh vocabulay built in cv
print("\nVocaboooo suda")
print(cv.vocabulary_)
## it prints huge vocbulary like this word: index, index is column number where that word is present
## reduce max_features from 2500 to 100 you can see all 100 words easily
"""
{'go': np.int64(870), 'point': np.int64(1579), 'crazi': np.int64(479), 'avail': np.int64(147), 
'bugi': np.int64(310), 'great': np.int64(904), 'world': np.int64(2433), 'la': np.int64(1164), 
'cine': np.int64(401), 'got': np.int64(889), 'wat': np.int64(2366), 'ok': np.int64(1476), 
'lar': np.int64(1172), 'joke': np.int64(1128), 'wif': np.int64(2401), 'oni': np.int64(1484), 
'free': np.int64(803), 'entri': np.int64(675), 'wkli': np.int64(2422), 'comp': np.int64(435), 
'win': np.int64(2405), 'fa': np.int64(716), 'cup': np.int64(493), 'final': np.int64(760), 
'tkt': np.int64(2214), 'st': np.int64(2001), 'may': np.int64(1311), 'text': np.int64(2156), 
'receiv': np.int64(1686), 'question': np.int64(1649), 'std': np.int64(2017), 'txt': np.int64(2271), 
'rate': np.int64(1667), 'appli': np.int64(103), 'dun': np.int64(617), 'say': np.int64(1776),
"""

## next lets apply ngram value (combination of words considered for vectorizing)
cv = CountVectorizer(max_features=100, binary=True, ngram_range=(1,1)) # top 100 frequently occurring words
X = cv.fit_transform(corpus).toarray() # apply to corpus to get your vector
print("\nVocaboooo - with ngram - (1,1)")
print(cv.vocabulary_) # shows
# {'go': np.int64(22), 'great': np.int64(25), 'got': np.int64(24), 'wat': np.int64(90), 'ok': np.int64(56), 'free': np.int64(18),

# next ngram - 1,2
cv = CountVectorizer(max_features=200, binary=True, ngram_range=(1,2)) # top 100 frequently occurring words
X = cv.fit_transform(corpus).toarray() # apply to corpus to get your vector
print("\nVocaboooo - with ngram - (1,2)")
print(cv.vocabulary_) # o/p at teh end you see double words combination too
#Ex: 'pleas call': np.int64(128)
## as u increse max_features you see more double words

## next try 2,2, you see all frequently occureing double words
cv = CountVectorizer(max_features=100, binary=True, ngram_range=(2,2)) # top 100 frequently occurring words
X = cv.fit_transform(corpus).toarray() # apply to corpus to get your vector
print("\nVocaboooo - with ngram - (2,2)")
print(cv.vocabulary_)
# {'free entri': np.int64(31), 'claim call': np.int64(16), 'call claim': np.int64(3), 'free call': np.int64(30), 'call mobil': np.int64(9),

## 3,3 words
#{'chanc win cash': np.int64(15), 'like lt gt': np.int64(44), 'sorri call later': np.int64(81), 'pleas call custom': np.int64(68), 'call custom servic': np.int64(6),
print(X)