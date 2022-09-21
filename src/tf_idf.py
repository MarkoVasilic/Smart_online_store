from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

'''Embedding data, transforming words into vectors'''
def tf_idef_emmbeding(train_data, whole_data):
    '''Making object od model'''
    TV = TfidfVectorizer(min_df=5, max_features=4000, token_pattern=r'\w{1,}', ngram_range=(1,2), sublinear_tf=1)
    '''Training model'''
    TV.fit(whole_data)
    '''Normalize data'''
    return normalize(TV.transform(train_data).toarray())
