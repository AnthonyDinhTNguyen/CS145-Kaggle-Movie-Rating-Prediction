
import scipy.spatial
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time

def cur_time_millis():
    return time.time()*1000.0

def print_t(msg):
    print(msg + " (time " + str(cur_time_millis()) + ")")

def getDistanceUsers(userA, userB, ratings_matrix):
    distance = scipy.spatial.distance.cosine(ratings_matrix.iloc[userA].values.tolist(), ratings_matrix.iloc[userB].values.tolist())
    return distance

print_t("Reading in movies.csv...")
df_m = pd.read_csv('../movies.csv', sep=',', names=['movieId', 'title', 'genres'], skiprows=1)

print_t("Reading in val_ratings_binary.csv...")
df_vrb = pd.read_csv('../val_ratings_binary.csv', sep=',', names=['userId', 'movieId', 'rating'], skiprows=1)

print_t("Combining movies and val_ratings_binary tables...")
df_combined = pd.merge(df_m, df_vrb, on='movieId')
df_combined.set_index('userId', inplace=True)
df_combined.sort_index(inplace=True)
df_combined.groupby(df_combined['userId','genres']).aggregate()

print(df_combined)

#print_t("Pivoting combined table...")
#df_combined.pivot(index='userId', columns='genres', values='rating')

#print(df_combined)
