
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

print("df_m head:")
print(df_m.head())

print("df_m describe:")
print(df_m.describe())

print("Number of unique genre combinations: " + str(df_m['genres'].nunique()))

print_t("Reading in val_ratings_binary.csv...")
df_vrb = pd.read_csv('../val_ratings_binary.csv', sep=',', names=['userId', 'movieId', 'rating'], skiprows=1)
print_t("Done reading in val_ratings_binary.csv.")

print_t("Replacing 0 with -1...")
df_vrb['rating'].replace(0, -1, inplace=True)

print("df_vrb head:")
print(df_vrb.head())

print("df_vrb describe:")
print(df_vrb.describe())

print_t("Pivoting user ratings data...")
ratings_matrix = df_vrb.pivot(index='userId', columns='movieId', values='rating')
print_t("Done pivoting user ratings data.")

print_t("Replacing NaN with 0 in pivoted user data...")
ratings_matrix.replace(np.nan, 0, inplace=True)

print(getDistanceUsers(0, 0, ratings_matrix))
