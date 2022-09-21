from transformers import pipeline
import pandas as pd
import csv
import sys
import argparse
sys.path.insert(0, '../src')

def load_data(num_of_rows):
    '''Load data'''
    df = pd.read_csv("../data/smart_store_reviews.csv")
    '''Randomize rows'''
    df = df.sample(frac=1).reset_index(drop=True)
    return df.iloc[:num_of_rows, :]


def bart(num_of_rows):
    '''Define model'''
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    '''Load data'''
    df = load_data(num_of_rows)
    summarized_reviews = []
    '''Get summary for each review'''
    for r in df['review_body']:
        summarized_reviews.append(summarizer(r, max_length=200, min_length=10, do_sample=False))
    df['summarized_review'] = [summary[0]["summary_text"] for summary in summarized_reviews]
    '''Save to file'''
    save_file(df)

def do_summarization(text):
    '''Define model'''
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    '''Get summary'''
    summary = summarizer(text, max_length=200, min_length=10, do_sample=False)[0]['summary_text']
    print(summary)
    return summary

def save_file(df):
    '''Open file'''
    with open('../data/data_with_summary.csv', 'w', encoding='UTF8', newline='') as f:
        '''Define writer'''
        writer = csv.writer(f)
        '''Write the header'''
        writer.writerow(df.columns.to_list())
        '''Write the data'''
        for index, row in df.iterrows():
            writer.writerow(row.values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Write number after script name for selecting option you want to use: | (ER) summarizing of existing reviews | | (IT) summarizing input text | *** and text for summary | *** and number of rows"
    )
    parser.add_argument('selection', help="Provide selection!")
    parser.add_argument('text', help="Provide text for summary!")
    parser.add_argument('num_of_rows', help="Provide number of rows!", type=int)
    args = parser.parse_args()
    if args.num_of_rows < 0 or args.num_of_rows > 208000:
        print("Error, wrong input for number of rows!")
    elif args.selection.upper() == 'ER':
        print("You selected summarizing existing reviews!")
        bart(args.num_of_rows)
    elif args.selection.upper() == 'IT':
        print("You selected to input your own text!")
        do_summarization(args.text)
    else:
        print("Error, something went wrong!")