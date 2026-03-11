#Tokenization using NLTK (Natural Language Toolkit) package ( or opensource- Spacy lib)
# Tokenization = converting corpus(paragraphs) in to tokens(sentences) or
# Tokenization = converting sentences into words

#note: Google's tensorflow and Meta's Pytorch are libs for building deeplearning model,
# don't confuse with nltk and spacy libs- these are for natural lang processing - text processing libs

# compare NLTK and spaCY
# nltk is for learning nlp concepts, suitable for small dataset, not production grade, and is slower
# spaCY has industrial strength for production grade apps, faster, suitable for huge datasets, used in realworld nlp pipelines

#need >> pip install nltk==3.7 # other versions don't work

## 1. Tokenization - sentence tokenizer
## paragraph ->sentences

corpus = """Hello welcome to krish naik's NLP tutorials.
please watch the entire course! to become and expert in NLP
"""

from nltk.tokenize import sent_tokenize # sentence tokenizer
tokenizedSentences = sent_tokenize(corpus) ## tokenized corpus you see 3 sentences from 1 para-corpus ["Hello welcome to krish naik's NLP tutorials.", 'please watch the entire course!', 'to become and expert in NLP']
print(tokenizedSentences)
print(type(tokenizedSentences)) #list
for sent_token in tokenizedSentences:
    print(sent_token)

"""output: for tokenized sentences 
Hello welcome to krish naik's NLP tutorials.
please watch the entire course!
to become and expert in NLP
"""

## 2. Tokenization - word tokenizer
## paragraph ->words
## sentence -> words

from nltk.tokenize import word_tokenize
word_tokens=word_tokenize(corpus) # word tokenizer
print(word_tokens)
for word in word_tokens:
    print(word) # you see list of words printed

# another word tokenizer - wordpunct_tokenize
from nltk.tokenize import wordpunct_tokenize
wordpunct_tokens = wordpunct_tokenize(corpus) # punctuations are considered as separate words
print(wordpunct_tokens)

# another word tokenizer
from nltk.tokenize import TreebankWordTokenizer
Treebankwordtokens = TreebankWordTokenizer().tokenize(corpus)
print(f"Tree bank word tokens: {Treebankwordtokens}")

