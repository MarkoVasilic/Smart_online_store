from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''Embedding data, transforming words into vectors'''
def word2vec_emmbeding(data):
    '''We have to split sentences into array of words because word2vec works that way'''
    new_data = []
    for d in data:
        new_data.append(d.split(" "))
    '''Making object od model and training it'''
    return Word2Vec(new_data, vector_size=300, window=5, min_count=2)

#df = pd.read_csv("../data/train_data.csv")
#data = df['review_body'].to_list()
#model = word2vec_emmbeding(data)
#print(model.wv.vectors.shape)
