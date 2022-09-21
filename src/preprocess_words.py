from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def text_preprocessing_eng(df, column_name):
    new_df = df.copy()    
    for index, row in new_df.iterrows():
        # split into tokens by white space
        tokens = word_tokenize(new_df.at[index, column_name])
        # make every word lowercase
        tokens = [w.lower() for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        stop_words.discard("not")
        words = [w for w in words if not w in stop_words]
        # stemming of words
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in words]
        # join words
        stemmed = ' '.join(stemmed)
        new_df.at[index, column_name] = stemmed
    return new_df