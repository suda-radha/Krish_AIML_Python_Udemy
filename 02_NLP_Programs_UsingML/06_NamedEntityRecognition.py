## Named Entity Recognition is -- identifying named entities (words) in sentence
## Ex Named entities could be Name, Place, Time, date, money, percentage, organization etc

sentence="The Eiffel Tower was built from 1887 to 1889 by French engineer Gustave Eiffel, whose company specialized in building metal frameworks and structures."

import nltk
nltk.download('words')
nltk.download('maxent_ne_chunker') ## needed as suggested by run failure message
words = nltk.word_tokenize(sentence)
pos_tags= nltk.pos_tag(words)
ne_tags=nltk.ne_chunk(pos_tags) ## tree output
print(type(ne_tags)) ## output is a tree data type = <class 'nltk.tree.tree.Tree'>
# so u can use a draw() method to see teh tree generated

print("Tokenized words: \n",words) # prints teh words (tokenized words)
print("Pos tags for words: \n",pos_tags) # prints the pos mapping for words
print("named entities for words: \n", ne_tags)

nltk.ne_chunk(pos_tags).draw()



