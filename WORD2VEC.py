import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re

paragraph = """All my friends. Settle down let me talk, I will get more and more emotional (crowd gets louder as he composes himself). 
My life, between 22 yards for 24 years, it is hard to believe that that wonderful journey has come to an end,
 but I would like to take this opportunity to thank people who have played an important role in my life.
 Also, for the first time in my life I am carrying this list, to remember all the names in case I forget someone. 
 I hope you understand. It's getting a little bit difficult to talk but I will manage.

The most important person in my life, and I have missed him a lot since 1999 when he passed away, my father.
 Without his guidance, I don't think I would have been standing here in front of you. 
 He gave me freedom at the age of 11, and told me that [I should] chase my dreams,
 but make sure you do not find shortcuts. The path might be difficult but don't give up,
 and I have simply followed his instructions. Above all, he told me to be a nice human being,
 which I will continue to do and try my best. Every time I have done something special [and] showed my bat, it was [for] my father."""



# Preprocessing data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the data
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


words = model.wv.vocab

# Finding Word Vectors
vector = model.wv['emotional']

# Most similar words
similar = model.wv.most_similar('best')