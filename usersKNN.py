
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

print_t("Reading in val_ratings_binary.csv...")
df = pd.read_csv('../val_ratings_binary.csv', sep=',', names=['userId', 'movieId', 'rating'], skiprows=1)
print_t("Done reading in val_ratings_binary.csv.")

print_t("Replacing 0 with -1...")
df['rating'].replace(0, -1, inplace=True)

print("df head:")
print(df)

print("df describe:")
print(df.describe())

print_t("Pivoting user ratings data...");
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
print_t("Done pivoting user ratings data.");

print_t("Replacing NaN with 0 in pivoted user data...")
ratings_matrix.replace(np.nan, 0, inplace=True)

print(getDistanceUsers(0, 0, ratings_matrix))
