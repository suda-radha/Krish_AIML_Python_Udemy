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
print("\n\n",corpus)

## let us create bag of words model (to build vector) using sklearn's countVectorizer
## refer documentation to see how to use that method
## choose max_features = choose top 2500 words that have max frequency or words repeated

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500, binary=True) # choose max_features = choose top 2500 words that have max frequency or words repeated
X = cv.fit_transform(corpus).toarray() # apply to corpus to get your vector
print("\n", X)
print(X.shape) #(5572, 2500)

## what we see is a normal BOW we can convert it to binary BOW -> binary=True


