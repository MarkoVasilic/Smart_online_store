import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import sys  
sys.path.insert(0, '../src')
import tf_idf
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import homogeneity_score
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
        '''Embedding data, transforming words into vectors'''
        return reduced_df['review_body'], reduced_df['stars']
    except:
        raise Exception("file_not_found")

def evaluate_k_means(num_of_rows, max_k, file_path):
    '''Loading data'''
    text, stars = load_data(num_of_rows, file_path)
    '''Data embedding'''
    text_emb = tf_idf.tf_idef_emmbeding(text, text)
    '''Create range of different number of clusters'''
    range_n_clusters = range(2, max_k)
    res = []
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 88 for reproducibility.
        '''Train model'''
        clusterer = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=500, n_init=10, random_state=88)
        clusterer.fit(text_emb)
        res.append(evaluate(text_emb, clusterer, n_clusters, stars.to_list()))
    return res

def evaluate(data, clusterer, n_clusters, original_labels):
    '''Predict data'''
    cluster_labels = clusterer.predict(data)
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    calinski_harabasz_index = calinski_harabasz_score(data, cluster_labels)
    print("Calinski-Harabasz Index: %0.3f" % calinski_harabasz_index)
    davies_bouldin_index = davies_bouldin_score(data, cluster_labels)
    print("Davies-Bouldin Index: %0.3f" % davies_bouldin_index)
    homogeneity_score_result = homogeneity_score(original_labels, cluster_labels)
    print("Homogeneity score is:  %0.3f" % homogeneity_score_result)
    return {"Num_of_clusters" : n_clusters, 
    "Average_silhouette_score" : silhouette_avg,
    "Calinski_Harabasz_Index" : calinski_harabasz_index,
    "Davies_Bouldin_Index" : davies_bouldin_index,
    "Homogeneity_score" : homogeneity_score_result}

def  k_means_elbow(num_of_rows, max_k, file_path):
    '''Loading data'''
    text, stars = load_data(num_of_rows, file_path)
    '''Data embedding'''
    text_emb = tf_idf.tf_idef_emmbeding(text, text)
    K = range(1, max_k)
    sum_of_squared_distances = []
    '''For each number of clusters create model'''
    for k in K:
        model = KMeans(n_clusters=k, init='k-means++', max_iter=500, n_init=1, verbose=10, random_state=88)
        model.fit(text_emb)
        sum_of_squared_distances.append(model.inertia_)
    '''Plot results'''
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum_of_squared_distances')
    plt.show()

def k_means_5_clusters(num_of_rows, file_path):
    '''Loading data'''
    text, stars = load_data(num_of_rows, file_path)
    '''Data embedding'''
    text_emb = tf_idf.tf_idef_emmbeding(text, text)
    '''Train model'''
    model = KMeans(n_clusters=5, init='k-means++', max_iter=500, n_init=10, verbose=1, random_state=88)
    model.fit(text_emb)
    '''Create dataframe of old and predicted stars'''
    new_list = [x+1 for x in model.labels_]
    list_of_tuples = list(zip(text, stars, new_list))
    new_df = pd.DataFrame(list_of_tuples, columns=['Text', 'Original Stars', 'New Stars'])
    '''Count number od stars in each cluster'''
    print(new_df['New Stars'].value_counts())
    print(new_df['Original Stars'].value_counts())
    '''Create matrix of compariosn between original stars and clusters'''
    compare_matrix = np.zeros((5, 5))
    for index, row in new_df.iterrows():
        compare_matrix[row['Original Stars'] - 1][row['New Stars'] - 1]+=1
    print(compare_matrix)
    print(compare_matrix.sum(axis=1))
    print(compare_matrix.sum(axis=0))
    '''Evaluate model'''
    return evaluate(text_emb, model, 5, stars.to_list())

def dbscan(num_of_rows, file_path):
    '''Loading data'''
    text, stars = load_data(num_of_rows, file_path)
    '''Data embedding'''
    text_emb = tf_idf.tf_idef_emmbeding(text, text)
    '''Train model'''
    dbscan_model = DBSCAN(eps=0.1, min_samples=50)
    dbscan_result = dbscan_model.fit_predict(text_emb)
    '''Evaluate model'''
    unique_clusters = np.unique(dbscan_result)
    print("Number of unique clusters: ", unique_clusters)
    silhouette_avg = silhouette_score(text_emb, dbscan_result)
    print("The average silhouette_score is :", silhouette_avg)
    return {"Unique_clusters" : len(unique_clusters),
    "Average_silhouette_score" : silhouette_avg}

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(
        description="Write number after script name for selecting algorithm you want to use: | (KM5) for Kmeans with 5 clusters | | (KM) for Kmeans with different number of clusters | | (KME) for Kmeans elbow | | (DBS) for DBSCAN | | *** and number of rows of data (1 - 208000) after that or 0 for all data | *** and max number of clusters (2 - 1000)"
    )
    parser.add_argument('selection', help="Provide selection!")
    parser.add_argument('num_of_rows', help="Provide number of rows!", type=int)
    parser.add_argument('num_of_max_clusters', help="Provide number of maximum clusters!", type=int)
    args = parser.parse_args()
    if args.num_of_rows < 0 or args.num_of_rows > 208000:
        print("Error, wrong input for number of rows!")
    elif args.num_of_max_clusters < 2 or args.num_of_max_clusters > 1000:
        print("Error, wrong input for number of clusters!")
    elif args.selection.upper() == 'KM5':
        print("You selected Kmeans with 5 clusters!")
        k_means_5_clusters(args.num_of_rows, "")
    elif args.selection.upper() == 'KM':
        print("You selected Kmeans with different number of clusters!")
        evaluate_k_means(args.num_of_rows, args.num_of_max_clusters, "")
    elif args.selection.upper() == 'KME':
        print("You selected Kmeans elbow!")
        k_means_elbow(args.num_of_rows, args.num_of_max_clusters, "")
    elif args.selection.upper() == 'DBS':
        print("You selected DBSCAN!")
        dbscan(args.num_of_rows, "")
    else:
        print("Error, wrong input for algorithm name!")
    end = time.time()
    print("The time of execution of above program is :", (end-start) / 60, "minutes")