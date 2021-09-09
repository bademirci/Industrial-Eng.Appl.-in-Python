import numpy as np
import pandas as pd

def getName():
    #TODO: Add your full name instead of Lionel Messi
    return "Batuhan Demirci"

def getStudentID():
    #TODO: Replace X's with your student ID. It should stay as a string and should have exactly 9 digits in it.
    return "070190155"

#Please use this function to initial cluster assignment. DO NOT WRITE your own function.
def initial_centroids(X,number_cluster):
    centroid_indexes=np.random.choice(range(len(X)),number_cluster,replace=False)
    centroids=X[centroid_indexes]
    return np.array(centroids)
    

#You can also define your own functions here if necessary 
def manhattan_distance(X1, X2):
    return (np.sum(np.absolute(X1 - X2)))


def assign_clusters(X,centroids):
    clusters = []
    for i in range(len(X)):
        distances = []
        for centroid in centroids:
            distances.append(manhattan_distance(centroid,X[i,:]))
        cluster = distances.index(min(distances))
        clusters.append(cluster)
    return np.array(clusters)

def calc_centroids(X,clusters):
    new_centroids = []
    for c in set(clusters):
        current_cluster = X[clusters==c]
        cluster_median = np.median(X[clusters==c],axis=0)
        new_centroids.append(cluster_median)
    return np.array(new_centroids)

def calc_total_variance(X,clusters,centroids):
    Wks=[]
    for c in set(clusters):
        current_cluster = X[clusters==c]
        Wk=0
        for i in range(len(current_cluster)):
            Wk=Wk+sum(abs(current_cluster[i,:]-centroids[c,:]))
        Wks.append(Wk)
    return np.sum(Wks)

def k_median_clustering(X,number_cluster,replication_number,epsilon):
    np.random.seed(42) #Do not change the seed
    best_cost = float('inf')
    for i in range(replication_number):
        difference = 1000
        centroids = initial_centroids(X, number_cluster)
        clusters = assign_clusters(X, centroids)
        variance = calc_total_variance(X, clusters, centroids)
        while difference > epsilon:
            centroids = calc_centroids(X, clusters)
            clusters = assign_clusters(X, centroids)
            newvariance = calc_total_variance(X, clusters, centroids)
            difference = np.abs(variance - newvariance)
            variance = newvariance
        if variance <= best_cost:
            best_cluster = clusters
            best_centroid = centroids
            best_cost = variance
    return best_cost, best_cluster, best_centroid



