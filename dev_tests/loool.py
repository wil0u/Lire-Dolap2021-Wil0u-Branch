import numpy as np
from sklearn.metrics import pairwise_distances

def make_double_black_white_box_slice(weights1,weights2,X,dist,threshold):

    return np.where(dist < threshold, np.dot(X, weights1), np.dot(X, weights2))



weights1 = [0,0,2,2]
weights2 = [4,4,0,0]

dist = np.asarray([0.8,0.7,0.3,0.1])

x_user = [2,2,1,0]

X = [[1,1,3,3] , [5,5,1,1], [1,1,1,1], [2,3,4,5]]

dist_ = pairwise_distances(np.asarray([x_user]),X)
threshold = 4

y = make_double_black_white_box_slice(weights1,weights2,X,dist_,threshold)
print()