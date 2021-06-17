from utility import read_sparse
import numpy as np
from scipy.spatial import distance
import random
from scipy.stats import truncnorm
all_actual_ratings, iid_map = read_sparse("./ml-latest-small/ratings.csv")
samples = all_actual_ratings.toarray()
from sklearn.neighbors import NearestNeighbors
import sklearn

class NormalPredictor:
    def __init__(self,low,up):
        self.low = low
        self.up = up
    def fit(self,X):
        self.X = X
        self.mean = np.mean(X[np.where(X > 0)])
        self.std = np.std(X[np.where(X > 0)])
        self.truncNorm = truncnorm(
            (self.low - self.mean) / self.std, (self.up - self.mean) / self.std, loc=self.mean, scale=self.std)

    def predict(self,x,i):

        return self.truncNorm.rvs()

    def predictSlice(self,X,i):
        return np.asarray(self.truncNorm.rvs(len(X)))

    def predictForAllItem(self,x):
        return np.asarray(self.truncNorm.rvs(len(x)))

'''
normalPred = NormalPredictor(low=0,up=5)
normalPred.fit(samples)

X = [1,3,4,5,5]
test = normalPred.predictSlice(X)
for i in range(15):
    print(normalPred.predict())

'''