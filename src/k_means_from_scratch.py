import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn.datasets import make_blobs  
sys.path.insert(0, '../src')
import tf_idf
from sklearn.metrics import silhouette_score
import argparse
import time

np.random.seed(42)

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


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [
            euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)
        ]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

def k_means_from_scratch(num_of_rows, num_of_max_clusters, file_path):
    data, stars = load_data(num_of_rows, file_path)
    data_emb = tf_idf.tf_idef_emmbeding(data, data)
    k = KMeans(K=num_of_max_clusters, max_iters=150, plot_steps=False)
    y_pred = k.predict(data_emb)
    silhouette_avg = silhouette_score(data_emb, y_pred)
    print("For n_clusters =", num_of_max_clusters, "The average silhouette_score is :", silhouette_avg)
    return {"Num_of_clusters" : num_of_max_clusters,
    "Average_silhouette_score" : silhouette_avg}

# Testing
if __name__ == "__main__":
    what = -1
    start = time.time()
    parser = argparse.ArgumentParser(
        description="Write code after script name for selecting algorithm you want to use: | (TD) for Text data | | (ND) for Number data | | *** and number of rows of data (1 - 208000) after that or 0 for all data | *** and max number of clusters (2 - 1000)"
    )
    parser.add_argument('selection', help="Provide selection!")
    parser.add_argument('num_of_rows', help="Provide number of rows!", type=int)
    parser.add_argument('num_of_max_clusters', help="Provide number of maximum clusters!", type=int)
    args = parser.parse_args()
    if args.num_of_rows < 0 or args.num_of_rows > 208000:
        print("Error, wrong input for number of rows!")
    elif args.num_of_max_clusters < 2 or args.num_of_max_clusters > 1000:
        print("Error, wrong input for number of clusters!")
    elif args.selection.upper() == 'TD':
        print("You selected text data!")
        what = 0
    elif args.selection.upper() == 'ND':
        print("You selected number data!")
        what = 1
    else:
        print("Error, wrong input for algorithm name!")
    
    if what == 1:
        X, y = make_blobs(
            centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
        )
        print(X.shape)

        clusters = len(np.unique(y))
        print(clusters)

        k = KMeans(K=clusters, max_iters=150, plot_steps=True)
        y_pred = k.predict(X)
        k.plot()
    elif what == 0:
        k_means_from_scratch(args.num_of_rows, args.num_of_max_clusters, "")
    end = time.time()
    print("The time of execution of above program is :", (end-start) / 60, "minutes")