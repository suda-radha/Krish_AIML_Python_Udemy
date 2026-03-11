## not all words are reduced using stemming, some words produced are really bad, ex: history---->histori
## lemmatizations can be used instead of stemming
## lemmatization uses a dictionaly of words to refer to

## NLTK provides WordNetLemmatizer class which is a thin wrapper around teh wordnet corpus(kind of dictionary of all words)
## Use cases - Q&A, chatbot, text summarization - used in many companies

## wordnet Lemmatizer
## output is called lemma - the root word (in previous case it ws stem word)
import nltk
nltk.download('wordnet') ## you need this dataset to be downloaded else u get error
nltk.download('omw-1.4') ## something that is recommended

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmtized_word = lemmatizer.lemmatize("going")
print(lemmtized_word) ## shows op=going

## using pos tag (to say if teh word in n-noun, v-verb, a-adjective, r-adverb
lemmtized_word2 = lemmatizer.lemmatize("going", pos='n')
print(lemmtized_word2) # op=going

lemmtized_word3 = lemmatizer.lemmatize("going", pos='v')
print(lemmtized_word3) # op=go (as expected)

## lemmatizing a bunch of words with default pos='n'
words= ["eating", "eats", "eaten", "writing", "writes", "programs", "programming", "history", "finally", "finalized"]
for word in words:
    print(word+"-->"+lemmatizer.lemmatize(word))

""" output 
eating-->eating
eats-->eats
eaten-->eaten
writing-->writing
writes-->writes
programs-->program
programming-->programming
history-->history
finally-->finally
finalized-->finalized
"""

## lemmatizing a bunch of words with pos=v
words= ["eating", "eats", "eaten", "writing", "writes", "programs", "programming", "history", "finally", "finalized"]
for word in words:
    print(word+"-->"+lemmatizer.lemmatize(word, pos='v'))

""" output
eats-->eat
eaten-->eat
writing-->write
writes-->write
programs-->program
programming-->program
history-->history
finally-->finally
finalized-->finalize
"""

print(lemmatizer.lemmatize("goes",pos='v')) ## go
print(lemmatizer.lemmatize("sportingly",pos='v')) ## sportingly
print(lemmatizer.lemmatize("fairly",pos='v')) ## fairly
