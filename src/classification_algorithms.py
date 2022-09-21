import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import sys  
sys.path.insert(0, '../src')
import tf_idf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse
import time

def load_data(num_of_samples, file_path):
    try:
        '''Function for loading data'''
        if file_path == "":
            df = pd.read_csv("../data/cleaned_data.csv")
        else:
            df = pd.read_csv(file_path)
        '''Shuffling data because it's sorted by stars'''
        df = df.sample(frac=1).reset_index(drop=True)
        '''Selecting required number of rows'''
        if num_of_samples != 0 and num_of_samples < len(df):
            reduced_df = df.iloc[:num_of_samples, :]
        else:
            reduced_df = df
        '''Transforming words into vectors'''
        return reduced_df['review_body'], reduced_df['stars']
    except:
        raise Exception("file_not_found")

def split_data(text, stars):
    '''Spliting data to train and test data'''
    return train_test_split(text, stars, test_size=0.2, random_state=42, shuffle=True)

def scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

def evaluate_results(model, X_test, Y_test):
    '''Calculating accuracy of model'''
    nr_correct = (Y_test == model.predict(X_test)).sum()
    print(f'{nr_correct} documents classified correctly')
    nr_incorrect = Y_test.size - nr_correct
    print(f'{nr_incorrect} documents classified incorrectly')
    classifier_score = model.score(X_test, Y_test)
    print(f'Classifier score is: {classifier_score}')
    '''Calculating classification report'''
    classification_report_result = classification_report(Y_test, model.predict(X_test), target_names=['1', '2', '3', '4', '5'])
    print(classification_report_result)
    return {"documents_classified_correctly": float(nr_correct),
     "documents_classified_incorrectly": float(nr_incorrect),
     "classifier_score" : float(classifier_score),
     "classification_report": classification_report_result}

'''Naive bayes multinominal algorithm'''
def multinomial_NB(num_of_rows, file_path):
    '''Loading data'''
    text, stars = load_data(num_of_rows, file_path)
    '''Spliting data'''
    X_train, X_test, Y_train, Y_test = split_data(text, stars)
    '''Embedding data'''
    X_train_emb = tf_idf.tf_idef_emmbeding(X_train, text)
    X_test_emb = tf_idf.tf_idef_emmbeding(X_test, text)
    '''Making object od model'''
    classifier = MultinomialNB(alpha=0.6)
    '''Training model'''
    classifier.fit(X_train_emb, Y_train)
    '''Checking accuracy'''
    return evaluate_results(classifier, X_test_emb, Y_test)

'''Random forest algorithm'''
def random_forest(num_of_rows, file_path):
    '''Loading data'''
    text, stars = load_data(num_of_rows, file_path)
    '''Spliting data'''
    X_train, X_test, Y_train, Y_test = split_data(text, stars)
    '''Embedding data'''
    X_train_emb = tf_idf.tf_idef_emmbeding(X_train, text)
    X_test_emb = tf_idf.tf_idef_emmbeding(X_test, text)
    '''Making object od model'''
    # n_estimator - number of trees used
    # verbose - printing status of training
    classifier = RandomForestClassifier(verbose=1, n_estimators=70)
    '''Training model'''
    classifier.fit(X_train_emb, Y_train)
    '''Checking accuracy'''
    return evaluate_results(classifier, X_test_emb, Y_test)

'''Logistic regression algorithm'''
def logistic_regresion(num_of_rows, file_path):
    '''Loading data'''
    text, stars = load_data(num_of_rows, file_path)
    '''Spliting data'''
    X_train, X_test, Y_train, Y_test = split_data(text, stars)
    '''Embedding data'''
    X_train_emb = tf_idf.tf_idef_emmbeding(X_train, text)
    X_test_emb = tf_idf.tf_idef_emmbeding(X_test, text)
    '''Making object od model'''
    # solver - Algorithm to use in the optimization problem. 
    # max_iter - Maximum number of iterations taken for the solvers to converge.
    # multi_class for number of classes in our data (binary or more)
    # class weight - The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
    classifier = LogisticRegression(solver='newton-cg', verbose=1, max_iter=1000, multi_class='multinomial', class_weight='balanced')
    '''Training model'''
    classifier.fit(X_train_emb, Y_train)
    '''Checking accuracy'''
    Y_pred = classifier.predict_proba(X_test_emb)
    score = classifier.score(X_test_emb, Y_test)
    print(score)
    return evaluate_results(classifier, X_test_emb, Y_test)

def support_vector_machines(num_of_rows, file_path):
    '''Loading data'''
    text, stars = load_data(num_of_rows, file_path)
    '''Scaling data'''
    #text_scaled = scaler(text)
    '''Spliting data'''
    X_train, X_test, Y_train, Y_test = split_data(text, stars)
    '''Embedding data'''
    X_train_emb = tf_idf.tf_idef_emmbeding(X_train, text)
    X_test_emb = tf_idf.tf_idef_emmbeding(X_test, text)
    '''Making object od model'''
    classifier = svm.SVC(verbose=1, gamma='scale', class_weight='balanced', decision_function_shape='ovo', random_state=88)
    '''Training model'''
    classifier.fit(X_train_emb, Y_train)
    '''Checking accuracy'''
    return evaluate_results(classifier, X_test_emb, Y_test)


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(
        description="Write number after script name for selecting algorithm you want to use: | (MNB) for MultinomialNB | | (RF) for Random Forest | | (LR) for Logistic Regression | | (SVM) for Support Vector Machines | *** and number of rows of data (1 - 208000) after that or 0 for all data"
    )
    parser.add_argument('selection', help="Provide selection!")
    parser.add_argument('num_of_rows', help="Provide number of rows!", type=int)
    args = parser.parse_args()
    if args.num_of_rows < 0 or args.num_of_rows > 208000:
        print("Error, wrong input for number of rows!")
    elif args.selection.upper() == 'MNB':
        print("You selected MultinomialNB algorithm!")
        multinomial_NB(args.num_of_rows, "")
    elif args.selection.upper() == 'RF':
        print("You selected Random Forest algorithm!")
        random_forest(args.num_of_rows, "")
    elif args.selection.upper() == 'LR':
        print("You selected Logistic Regression algorithm!")
        logistic_regresion(args.num_of_rows, "")
    elif args.selection.upper() == 'SVM':
        print("You selected Support Vector Machines algorithm!")
        support_vector_machines(args.num_of_rows, "")
    else:
        print("Error, wrong input for algorithm name!")
    end = time.time()
    print("The time of execution of above program is :", (end-start) / 60, "minutes")