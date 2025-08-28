import math
import sys
import numpy as np

def get_squared_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2) 

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    # Your code here
    if (len(initial_centroids) < k):
        return []
    
    pts = np.array(points)
    final_centroids = np.array(initial_centroids[0:k], dtype=float)
    
    for _ in range(max_iterations):
        
        clusters = [[] for _ in range(k)]
        for pt in pts:
            c_id = 0
            minSqDist = sys.float_info.max
            for i, centroid in enumerate(final_centroids):
                sqDist = get_squared_distance(pt, centroid)
                if sqDist < minSqDist:
                    minSqDist = sqDist
                    c_id = i
                    
            clusters[c_id].append(pt)
            
        for c_id, cluster in enumerate(clusters):
            npCluster = np.array(cluster)
            final_centroids[c_id] = ((final_centroids[c_id] + npCluster.mean(axis=0)) / 2).round(decimals=4)
    
    return final_centroids

points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
k = 2
initial_centroids = [(1, 1), (10, 1)]
max_iterations = 10
print(k_means_clustering(points, k, initial_centroids, max_iterations))