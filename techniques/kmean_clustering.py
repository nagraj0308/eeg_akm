from sklearn.cluster import KMeans


def k_mean_clustering(data):
    kmeans = KMeans(2, random_state=0)
    kmeans.fit(data)
    result = kmeans.fit_predict(data)
    return result
