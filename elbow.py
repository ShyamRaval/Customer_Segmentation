from kneed import KneeLocator
from sklearn.cluster import KMeans

def get_optimum_k(df):
    #i) find WCSS
    wcss = []
    k_values = range(1, 11)

    for i in k_values:
        model = KMeans(n_clusters=i)
        model.fit(df)
        wcss.append(model.inertia_)

    #ii) find the optimum k using KneeLocator
    KL = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")

    return KL.elbow