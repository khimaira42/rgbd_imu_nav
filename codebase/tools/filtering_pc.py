import numpy as np
from sklearn.cluster import DBSCAN

def cluster_and_filter_points(points,size_threshold):
    """
    Cluster point cloud and remove clusters smaller than size_threshold.
    
    :param points: Array of points (numpy array of shape (n_points, 3)).
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    :param size_threshold: Minimum cluster size to keep.
    :return: Filtered array of points.
    """
    # Parameters for DBSCAN
    eps = 0.1  # Maximum distance between two samples
    min_samples = 30  # Minimum number of samples in a neighborhood

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    # Identify indices of points in clusters larger than the threshold
    large_cluster_indices = np.where(np.bincount(labels + 1) > size_threshold)[0] - 1  # -1 to account for noise label (-1)

    # Filter points belonging to large clusters
    filtered_points = points[np.isin(labels, large_cluster_indices)]
    
    return filtered_points
