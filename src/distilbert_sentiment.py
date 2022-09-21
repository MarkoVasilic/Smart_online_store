import pandas as pd
from transformers import pipeline
import csv
import argparse

def load_data(num_of_rows):
    '''Load data'''
    df = pd.read_csv("../data/cleaned_data.csv")
    '''Randomize rows'''
    df = df.sample(frac=1).reset_index(drop=True)
    return df.iloc[:num_of_rows, :]

def check_sentiment(eval, star):
    if (star < 3 and eval != 'neg') or (star > 3 and eval != 'pos') or (star == 3 and eval != 'neu'):
        return False
    return True

def calculate_score(df):
    distilbert_score = []
    for index, row in df.iterrows():
        distilbert_score.append(check_sentiment(row['distilbert_sentiment'][-3:], row['stars']))
    df['distilbert_score'] = distilbert_score
    print(sum(distilbert_score)/len(df))

def distilbert(num_of_rows):
    '''Define model'''
    classifier = pipeline("sentiment-analysis", model = "distilbert-base-uncased-finetuned-sst-2-english")
    '''Load data'''
    df = load_data(num_of_rows)
    '''Get summary for each review'''
    distilbert_sentiment = []
    for index, row in df.iterrows():
        res = classifier(row['review_body'])[0]
        if res['score'] > 0.7 and res['label'] == 'POSITIVE':
            distilbert_sentiment.append('distilbert_pos')
        elif res['score'] > 0.7 and res['label'] == 'NEGATIVE':
            distilbert_sentiment.append('distilbert_neg')
        else:
            distilbert_sentiment.append('distilbert_neu')
    df['distilbert_sentiment'] = distilbert_sentiment
    calculate_score(df)
    save_file(df)

def do_sentiment(text):
    '''Define model'''
    classifier = pipeline("sentiment-analysis", model = "distilbert-base-uncased-finetuned-sst-2-english")
    '''Get sentiment'''
    res = classifier(text)[0]
    print(res)
    ret = ""
    if res['score'] > 0.7 and res['label'] == 'POSITIVE':
        ret = 'Positive'
    elif res['score'] > 0.7 and res['label'] == 'NEGATIVE':
        ret = 'Negative'
    else:
        ret = 'Neutral'
    print(ret)
    return ret

def save_file(df):
    '''Open file'''
    with open('../data/data_with_distilbert_sentiment.csv', 'w', encoding='UTF8', newline='') as f:
        '''Define writer'''
        writer = csv.writer(f)
        '''Write the header'''
        writer.writerow(df.columns.to_list())
        '''Write the data'''
        for index, row in df.iterrows():
            writer.writerow(row.values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Write number after script name for selecting option you want to use: | (ER) sentiment of existing reviews | | (IT) sentiment input text | *** and text for sentiment | *** and number of rows"
    )
    parser.add_argument('selection', help="Provide selection!")
    parser.add_argument('text', help="Provide text for sentiment!")
    parser.add_argument('num_of_rows', help="Provide number of rows!", type=int)
    args = parser.parse_args()
    if args.num_of_rows < 1 or args.num_of_rows > 208000:
        print("Error, wrong input for number of rows!")
    elif args.selection.upper() == 'ER':
        print("You selected to sentiment existing reviews!")
        distilbert(args.num_of_rows)
    elif args.selection.upper() == 'IT':
        print("You selected to input your own text!")
        do_sentiment(args.text)
    else:
        print("Error, something went wrong!")