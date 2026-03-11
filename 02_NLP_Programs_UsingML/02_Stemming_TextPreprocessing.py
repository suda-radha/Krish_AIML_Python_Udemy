## Stemming --> is reducing a word to its root word/stem word --> having them as they are is not going to impact the o/p much,
# so it is ok to reduce the number of words
## Ex: goes, gone, going----> go is the stem word for all these words
## Ex: eating, eat, eaten ---> eat is teh stem word

##Practical use of stemming --> for classification problem weather email is spam or not

## 3 types of stemming: Porter stemming, Regexp stemming, Snowball stemming

## Stemming using nltk
## 1. PorterStemmer

## major disadvantage of stemming is it does not stem out all words, sometime it doesn't give proper stem words as seen below
## this issue will be fixed in lementing concept later

words= ["eating", "eats", "eaten", "writing", "writes", "programs", "programming", "history", "finally", "finalized"]
from nltk.stem import PorterStemmer
for word in words:
    print(word+"---->"+PorterStemmer().stem(word))
""" output shows as:
eating---->eat
eats---->eat
eaten---->eaten
writing---->write
writes---->write
programs---->program
programming---->program
history---->histori ### This is an issue with PorterStemmer()
finally---->final
finalized---->final
"""

print(PorterStemmer().stem("congratulations")) ## output is congratul which is invalid word

## 2. RegexpStemmer
## you can remove matching cases - see ex below, you remove ending words like able, ing, s, e

from nltk.stem import RegexpStemmer
regex_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)
print(regex_stemmer.stem('eating')) ## o/p =eat
print(regex_stemmer.stem('portable')) ## o/p = port


### 3. Snowball stemmer
# snowballs temmer is better than porter stemmer and regexstemmer
from nltk.stem import SnowballStemmer
SnowballStemmer('english')

for word in words:
    print(word+"--->"+SnowballStemmer('english').stem(word)) ## thsi alos shows history as histori

print(SnowballStemmer('english').stem("congratulations")) ## shows congratul

## Comparing porter stemmer and snowball stemmer
print(PorterStemmer().stem("fairly")), ## fairli
print(PorterStemmer().stem("sportingly")) ## sportingli
print(SnowballStemmer("english").stem("fairly")) ## fair
print(SnowballStemmer("english").stem("sportingly")) ## sport

## hence snowball stemmer is better than porter stemmer

