
import scipy.spatial
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time

def cur_time_millis():
    return time.time() * 1000.0

def print_t(msg):
    print(msg + " (time " + str(cur_time_millis()) + ")")

def getDistanceUsers(userA, userB, ratings_matrix):
    distance = scipy.spatial.distance.cosine(ratings_matrix.iloc[userA].values.tolist(), ratings_matrix.iloc[userB].values.tolist())
    return distance

def prepareUserInfo():
    print_t("Reading in movies.csv...")
    df_m = pd.read_csv('../movies.csv', sep=',', names=['movieId', 'title', 'genres'], skiprows=1)

    print_t("Description of movies.csv:")
    print(df_m.describe())

    print_t("Reading in val_ratings_binary.csv...")
    df_vrb = pd.read_csv('../val_ratings_binary.csv', sep=',', names=['userId', 'movieId', 'rating'], skiprows=1)

    print_t("Description of val_ratings_binary.csv:")
    print(df_vrb.describe())

    print_t("Combining movies and val_ratings_binary tables...")
    df_combined = pd.merge(df_m, df_vrb, on='movieId')
    df_combined.set_index('userId', inplace=True)
    df_combined.sort_index(inplace=True)
    df_combined[['rating']] = df_combined[['rating']].apply(pd.to_numeric)

    print_t("Before group by:")
    print(df_combined)

    aggregation_functions = {'rating': 'sum', 'movieId': 'first', 'title': 'first'}
    df_combined = df_combined.groupby(['userId','genres']).aggregate(aggregation_functions).reset_index()

    print_t("After group by:")
    print(df_combined)

    ratings_matrix = df_combined.pivot(index='userId', columns='genres', values='rating')
    ratings_matrix.replace(np.nan, 0, inplace=True)

    print_t("After pivot:")
    print(ratings_matrix)

    return ratings_matrix

def getKNNUsers(userId, ratings_matix, k):
    similarIDs = []
    i = 0
    while i < len(ratings_matix):
        currentUserId = i + 1
        print("Comparing " + str(userId) + " with " + str(currentUserId))
        if (currentUserId != userId):
            dist = getDistanceUsers(userId, currentUserId - 1, ratings_matrix)
            similarIDs.append([currentUserId, dist])
        i += 1
    similarIDs.sort(key = lambda x: x[1])
    return similarIDs[:k]

ratings_matrix = prepareUserInfo()

print(ratings_matrix)

print(getKNNUsers(0, ratings_matrix, 5))

print(ratings_matrix.iloc[3].values.tolist())
print(ratings_matrix.iloc[4].values.tolist())
