# word2vec is google's deep learning model that has been trained on 3Billion words with output vector size of 1X300 dim for each word
## you need gensim lib >> pip install gensim
## refer link : https://huggingface.co/fse/word2vec-google-news-300

import gensim
from gensim.models import Word2Vec, keyedvectors

## download the model
import gensim.downloader as api
wv = api.load('word2vec-google-news-300') ## this will take some time as size is 1662 MB
vec_king = wv['king'] ## give you vector of word 'king'

#print(vec_king) # op is a 1x300 vector
#print(vec_king.shape) # op = (300,)
#print(wv.most_similar('cricket'))
"""output with similar words and thier similarity value (0-1)
[('cricketing', 0.8372225761413574), ('cricketers', 0.8165745735168457), ('Test_cricket', 0.8094819188117981), 
('Twenty##_cricket', 0.8068488240242004), ('Twenty##', 0.7624265551567078), ('Cricket', 0.75413978099823), 
('cricketer', 0.7372578382492065), ('twenty##', 0.7316356897354126), ('T##_cricket', 0.7304614186286926), 
('West_Indies_cricket', 0.6987985968589783)]
"""
#print(wv.most_similar('happy'))
""" output
[('glad', 0.7408890724182129), ('pleased', 0.6632170677185059), 
('ecstatic', 0.6626912355422974), ('overjoyed', 0.6599286794662476), 
('thrilled', 0.6514049172401428), ('satisfied', 0.6437949538230896), 
('proud', 0.636042058467865), ('delighted', 0.627237856388092), 
('disappointed', 0.6269949674606323), ('excited', 0.6247665286064148)]
"""
#print(wv.similarity('hockey', 'sport')) # op= 0.47289255
vec= wv['king'] - wv['man'] +wv['woman'] # let us see if this results in queen
#print(vec) # its prints 1x300 dim vector
print(wv.most_similar(vec))
"""output:
[('king', 0.8449392318725586), ('queen', 0.7300517559051514), 
('monarch', 0.645466148853302), ('princess', 0.6156251430511475), 
('crown_prince', 0.5818676352500916), ('prince', 0.5777117609977722), 
('kings', 0.5613663792610168), ('sultan', 0.5376775860786438), 
('Queen_Consort', 0.5344247817993164), ('queens', 0.5289887189865112)]
"""
