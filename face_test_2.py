from scipy.spatial.distance import cosine
import numpy as np

features = [
    np.array([0.1, 0.2, 0.3]),
    np.array([100, 1.0, 1.0]),
    np.array([0.11, 0.21, 0.32]),
    np.array([0.09, 0.18, 0.29]),
]
distance = cosine(features[0], features[1])
print(distance)