from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
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
    roberta_score = []
    for index, row in df.iterrows():
        roberta_score.append(check_sentiment(row['roberta_sentiment'][-3:], row['stars']))
    df['roberta_score'] = roberta_score
    print(sum(roberta_score)/len(df))

def save_file(df):
    '''Open file'''
    with open('../data/data_with_roberta_sentiment.csv', 'w', encoding='UTF8', newline='') as f:
        '''Define writer'''
        writer = csv.writer(f)
        '''Write the header'''
        writer.writerow(df.columns.to_list())
        '''Write the data'''
        for index, row in df.iterrows():
            writer.writerow(row.values)

def roberta(num_of_rows):
    '''Define model'''
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    '''Load data'''
    df = load_data(num_of_rows)
    roberta_sentiment = []
    for index, row in df.iterrows():
        encoded_text = tokenizer(row['review_body'], return_tensors='pt', padding=True, truncation=True, max_length=512)
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
        }
        for key, value in scores_dict.items():
            if value == max(scores_dict['roberta_pos'], scores_dict['roberta_neg'], scores_dict['roberta_neu']):
                roberta_sentiment.append(key)
    df['roberta_sentiment'] = roberta_sentiment
    calculate_score(df)
    save_file(df)

def do_sentiment(text):
    '''Define model'''
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    '''Get sentiment'''
    encoded_text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
    'NEGATIVE' : scores[0],
    'NEUTRAL' : scores[1],
    'POSITIVE' : scores[2]
    }
    for key, value in scores_dict.items():
            if value == max(scores_dict['POSITIVE'], scores_dict['NEGATIVE'], scores_dict['NEUTRAL']):
                print(key)
                return key


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
        roberta(args.num_of_rows)
    elif args.selection.upper() == 'IT':
        print("You selected to input your own text!")
        do_sentiment(args.text)
    else:
        print("Error, something went wrong!")
