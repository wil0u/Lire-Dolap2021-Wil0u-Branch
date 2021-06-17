import pandas as pd
import numpy as np
import utility
from scipy.spatial import distance


def cosine_distance_on_common_features(u,v,w=None):
    common_items_rated = np.intersect1d(np.nonzero(u),np.nonzero(v))
    if(len(common_items_rated) <1 ):
        return 1

    v_on_common_features = v[common_items_rated]
    u_on_common_features = u[common_items_rated]


    return distance.cosine(u_on_common_features,v_on_common_features,w)


movies = pd.read_csv('ml-latest-small/movies.csv', sep=",", header=0)
all_actual_ratings = pd.read_csv('ml-latest-small/ratings.csv', sep=",", header=0)

all_actual_ratings, iid_map = utility.read_sparse('ml-latest-small/ratings.csv')

u1 = all_actual_ratings.getrow(3)
u2 = all_actual_ratings.getrow(5)


cosine_distance_on_common_features(u1.toarray()[0],u2.toarray()[0])

from sklearn.neighbors import NearestNeighbors
X, iid_map = utility.read_sparse('ml-latest-small/ratings.csv')
X = X.toarray()
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

#Les points du cluster de l'utilisateur
user_cluster = X[np.random.choice(X.shape[0], 47, replace=False)]
#on génère la vérité terrain pour les points dans le cluster de l'utilisateur
y_user_cluster = [0]*47

#Les points randoms en dehors du cluster
random_points_outside_user_cluster = X[np.random.choice(X.shape[0], 100, replace=False)]
#On génère la vérité terrain pour les points en dehors du cluster de l'utilisateur
y_random_points_outside_user_cluster = [1]*100


#On fusionne les deux vérités terrain pour n'en faire qu'une
y_user_cluster.extend(y_random_points_outside_user_cluster)
y_smote = np.asarray(y_user_cluster)

#On fait de même pour les points dans le cluster et en dehors
X_smote = np.vstack([user_cluster,random_points_outside_user_cluster])
#Normalement les points dans y_smote respectent l'ordre dans X_smote

#On fit
X_res, y_res = smote.fit_resample(X_smote, y_smote)

#Et on récupère que les points ayant la vérité terrain = 0 ( les points du cluster + les points créer par smote)
neighborhood = X_res[np.where(y_res == 0)]


neigh = NearestNeighbors(
    n_neighbors=20, algorithm='brute',
    metric=distance.cosine)
neigh.fit(X.toarray())
u1 = X.getrow(3)
u2 = X.getrow(5)
u1_and_u2_neighbors_and_distances = neigh.kneighbors([u1.toarray()[0],u2.toarray()[0]])


array_test = np.asarray([[1,0,0],[1,0,0],[0,1,0]])

i = 2
indices = np.where(array_test[:,i]>0)
print()
'''
trainingSet = []
for index, row in all_actual_ratings.iterrows():

    movieId = int(row['movieId'])
    movieDetails = movies[movies['movieId'] == movieId]
    movieGenres = str(movieDetails['genres'].values[0]).split("|")
    movieVector = [1/len(movieGenres) for genre in movieGenres]

    userId = int(row['userId'])
    userRatings = all_actual_ratings[all_actual_ratings['userId'] == userId]['rating'].values
    userMean = np.mean(userRatings)
    userStd = np.std(userRatings)
    userMin = np.min(userRatings)
    userMax = np.max(userRatings)
    userVector = [userMean,userStd,userMax,userMin]

    vector = movieVector+userVector

    trainingSet.append(vector)
    print(row['movieId'],row['rating'],str(movieDetails['genres'].values[0]).split("|"))
'''

genresConcatenated = []

for index, row in all_actual_ratings.iterrows():
    movieId = int(row['movieId'])
    movieDetails = movies[movies['movieId'] == movieId]
    movieGenres = str(movieDetails['genres'].values[0]).split("|")
    genresConcatenated.extend(movieGenres)

allUniqueGenres = list(set(genresConcatenated))
trainingSet = []
for index, row in all_actual_ratings.iterrows():

    movieId = int(row['movieId'])
    movieDetails = movies[movies['movieId'] == movieId]
    movieGenres = str(movieDetails['genres'].values[0]).split("|")
    movieVector = [1 if genre in movieGenres else 0 for genre in allUniqueGenres]

    userId = int(row['userId'])
    userRatings = all_actual_ratings[all_actual_ratings['userId'] == userId]['rating'].values
    userMean = np.mean(userRatings)
    userStd = np.std(userRatings)
    userMin = np.min(userRatings)
    userMax = np.max(userRatings)
    userVector = [userMean, userStd, userMax, userMin]

    rating = [row['rating']]
    vector = movieVector + userVector + rating

    trainingSet.append(vector)

userColumns = ['userMean','userStd','userMin','userMax']
df_trainingSet = pd.DataFrame(data=trainingSet, columns=allUniqueGenres+userColumns+['rating'])
df_trainingSet.to_csv('trainingSetVectors.csv')
print()