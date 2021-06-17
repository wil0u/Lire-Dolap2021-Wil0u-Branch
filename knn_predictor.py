import numpy as np
from scipy.spatial import distance
import random
from sklearn.neighbors import NearestNeighbors
import sklearn



def cosine_distance_on_common_features(u,v,w=None):
    common_items_rated = np.intersect1d(np.nonzero(u),np.nonzero(v))
    if(len(common_items_rated) <1 ):
        return 1

    v_on_common_features = v[common_items_rated]
    u_on_common_features = u[common_items_rated]


    return distance.cosine(u_on_common_features,v_on_common_features,w)

import time

class KnnBasic:
    def __init__(self,k):
        self.k = k
        self.metric = cosine_distance_on_common_features
        self.algorithm = 'brute'
    def fit(self,X):
        self.X = X
        self.global_mean = np.mean(X[np.where(X > 0)])
        self.user_means = [np.mean(user_row[np.where(user_row > 0)]) for user_row in X]
        self.neigh = NearestNeighbors(
            n_neighbors=len(X), algorithm=self.algorithm,
            metric=self.metric)
        self.neigh.fit(X)
        print()
    def getNeighbors(self,x):
        return self.neigh.kneighbors([x])

    def getIndicesThatRatedItem(self,i):
        indices = np.where(self.X[:,i] > 0)
        return indices[0]

    def predict(self,x,i):
        item_id = i
        begin = time.time()
        return_kneighbors = self.neigh.kneighbors([x])
        end = time.time()
        #print(f"Total runtime of the kneighbors instruction is {end - begin}")
        dist = return_kneighbors[0][0]
        indices = return_kneighbors[1][0]
        indicesThatHaveRatedI = self.getIndicesThatRatedItem(i)
        sum_weighted_ratings = 0
        sum_similarities = 0
        compteur = 0
        for i in range(len(indices)):
            if (indices[i] in indicesThatHaveRatedI and compteur < self.k):
                sim_u_v = 1 - dist[i]
                sum_weighted_ratings = sum_weighted_ratings + (sim_u_v * self.X[indices[i], item_id])
                sum_similarities = sum_similarities + sim_u_v
                compteur = compteur + 1

        if(sum_similarities <= 0):
            return self.global_mean

        return min(sum_weighted_ratings/sum_similarities,5)


    def predictForAllItem(self,x):
        number_of_items = len(x)
        begin = time.time()
        return_kneighbors = self.neigh.kneighbors([x])
        end = time.time()
        print(f"Total runtime of the kneighbors instruction is {end - begin}")
        dist = return_kneighbors[0][0]
        indices = return_kneighbors[1][0]
        x_return = []
        for item_id in range(number_of_items):

            indicesThatHaveRatedI = self.getIndicesThatRatedItem(item_id)
            sum_weighted_ratings = 0
            sum_similarities = 0
            compteur = 0
            for i in range(len(indices)):
                if (indices[i] in indicesThatHaveRatedI and compteur < self.k):
                    sim_u_v = 1 - dist[i]
                    sum_weighted_ratings = sum_weighted_ratings + (sim_u_v * self.X[indices[i], item_id])
                    sum_similarities = sum_similarities + sim_u_v
                    compteur = compteur + 1
            if (sum_similarities <= 0):
                 x_return.append(self.global_mean)
            else:
                x_return.append(min(sum_weighted_ratings/sum_similarities,5))

        return x_return


    def predictSlice(self,X,i):
        preds = []
        for x in X:
            preds.append(self.predict(x,i))
        return np.asarray(preds)




class KnnWithMeans:
    def __init__(self,k):
        self.k = k
        self.metric = cosine_distance_on_common_features
        self.algorithm = 'brute'
    def fit(self,X):
        self.X = X
        self.user_means = [np.mean(user_row[np.where(user_row > 0)]) for user_row in X]
        self.global_mean = np.mean(X[np.where(X > 0)])
        self.neigh = NearestNeighbors(
            n_neighbors=len(X), algorithm=self.algorithm,
            metric=self.metric)
        self.neigh.fit(X)

    def getNeighbors(self,x):
        return self.neigh.kneighbors([x])

    def getIndicesThatRatedItem(self,i):
        indices = np.where(self.X[:,i] > 0)
        return indices[0]


    def predict(self,x,i):
        item_id = i
        return_kneighbors = self.neigh.kneighbors([x])
        dist = return_kneighbors[0][0]
        indices = return_kneighbors[1][0]
        indicesThatHaveRatedI = self.getIndicesThatRatedItem(i)
        sum_weighted_ratings = 0
        sum_similarities = 0
        compteur = 0
        mu_u = np.mean(x[np.where(x > 0)])
        for i in range(len(indices)):
            if (indices[i] in indicesThatHaveRatedI and compteur < self.k):
                sim_u_v = 1 - dist[i]
                sum_weighted_ratings = sum_weighted_ratings + mu_u + (sim_u_v * (self.X[indices[i], item_id] - self.user_means[indices[i]]))
                sum_similarities = sum_similarities + sim_u_v
                compteur = compteur + 1

        if(sum_similarities <= 0):
            return self.global_mean
        return min(sum_weighted_ratings/sum_similarities,5)


    def predictForAllItem(self,x):
        number_of_items = len(x)
        mu_u = np.mean(x[np.where(x > 0)])
        return_kneighbors = self.neigh.kneighbors([x])
        dist = return_kneighbors[0][0]
        indices = return_kneighbors[1][0]

        x_return = []
        for item_id in range(number_of_items):
            indicesThatHaveRatedI = self.getIndicesThatRatedItem(item_id)
            sum_weighted_ratings = 0
            sum_similarities = 0
            compteur = 0

            for i in range(len(indices)):
                if (indices[i] in indicesThatHaveRatedI and compteur < self.k):
                    sim_u_v = 1 - dist[i]
                    sum_weighted_ratings = sum_weighted_ratings + mu_u + (sim_u_v * (self.X[indices[i], item_id] - self.user_means[indices[i]]))
                    sum_similarities = sum_similarities + sim_u_v
                    compteur = compteur + 1
            if (sum_similarities <= 0):
                 x_return.append(self.global_mean)
            else:
                x_return.append(min(sum_weighted_ratings/sum_similarities,5))

        return x_return


    def predictSlice(self,X,i):
        preds = []
        for x in X:
            preds.append(self.predict(x,i))
        return np.asarray(preds)





'''
knn = KnnWithMeans(k=50)
knn.fit(samples)

for i in range(15):
    for j in range(15):
        user_id = random.randint(0,600)
        item_id = i
        real_rating = all_actual_ratings[user_id,item_id]
        return_kneighbors = knn.getNeighbors(samples[user_id])
        dist = return_kneighbors[0][0]
        indices = return_kneighbors[1][0]
        x = knn.predictWithMeans(samples[user_id],item_id)
        print()

sum_ratings = 0
sum_similarities = 0
for i in range(len(indices)):
    if (indices[i] != user_id):

        sum_ratings = sum_ratings + all_actual_ratings[indices[i],item_id]
        sum_similarities = 1 - dist[i]
print()'''

