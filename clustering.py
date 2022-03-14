from community import best_partition
from sklearn.base import BaseEstimator
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import community


def jaccard(x, y):
    # type: (set, set) -> float
    """Compute the Jaccard similarity between two sets."""
    return len(x & y) / len(x | y)


def matrix_to_knn_graph(data, k_neighbors, metric, progress_callback=None):
    # We do k + 1 because each point is closest to itself, which is not useful
    if metric == "cosine":
        # Cosine distance on row-normalized data has the same ranking as
        # Euclidean distance, so we use the latter, which is more efficient
        # because it uses ball trees. We do not need actual distances. If we
        # would, the N * k distances can be recomputed later.
        data = data / np.linalg.norm(data, axis=1)[:, None]
        metric = "euclidean"
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric=metric).fit(data)
    nearest_neighbors = knn.kneighbors(data, return_distance=False)
    # Convert to list of sets so jaccard can be computed efficiently
    nearest_neighbors = list(map(set, nearest_neighbors))
    num_nodes = len(nearest_neighbors)

    # Create an empty graph and add all the data ids as nodes for easy mapping
    graph = nx.Graph()
    graph.add_nodes_from(range(len(data)))

    for idx, node in enumerate(graph.nodes):
        if progress_callback:
            progress_callback(idx / num_nodes)

        for neighbor in nearest_neighbors[node]:
            graph.add_edge(
                node,
                neighbor,
                weight=jaccard(
                    nearest_neighbors[node], nearest_neighbors[neighbor]),
            )
    return graph


class LouvainMethod(BaseEstimator):
    def __init__(self, k_neighbors=50, metric="l2", resolution=1.0, random_state=None):
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.resolution = resolution
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        # If we are given a table, we have to convert it to a graph first
        graph = matrix_to_knn_graph(X, metric=self.metric, k_neighbors=self.k_neighbors)
        return self.fit_graph(graph)

    def fit_graph(self, graph):
        partition = best_partition(graph, resolution=self.resolution, random_state=self.random_state)
        self.labels_ = np.fromiter(list(zip(*sorted(partition.items())))[1], dtype=int)
        return partition, self.labels_, graph


def cluster_louvain(pca_vec, k_neighbors=50, metric="l2", resolution=1, random_state=None):
    if k_neighbors < 0: 
        print("k_neighbors must be greater than 0.")
        raise ValueError
        
    louv = LouvainMethod(k_neighbors=k_neighbors, metric=metric, resolution=resolution, random_state=random_state)
    p, c, G = louv.fit(pca_vec) 
    return p,c,G


def get_modularity(p,G): 
    modularity = community.modularity(p, G)
    return modularity


def get_document_per_cluster(df:pd.DataFrame, percentage, maximum_doc_size:int) -> int:
    """
    Returns number of documents per cluster. df must have a column named 'cluster'. 
    """
    # perform simple checks
    if maximum_doc_size < 1:
        print("maximum_doc_size must be greater than 1.") 
        raise ValueError
    if percentage > 100 or percentage < 0: 
        print("percentage must be between 0 and 100.")
        raise ValueError

    min_cluster_size = np.inf
    min_cluster = 0 
    for cluster in set(df.cluster): 
        docs_per_cluster = len(df[df.cluster == cluster])
        if docs_per_cluster < min_cluster_size: 
            min_cluster_size = docs_per_cluster
            min_cluster = cluster

    print(f"Minimum documents in cluster is {min_cluster_size}, cluster {min_cluster}")
    doc_per_cluster = max(round(min_cluster_size*percentage,0),maximum_doc_size)
    return doc_per_cluster

