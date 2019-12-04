
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
df_combined[['rating']] = df_combined[['rating']].apply(pd.to_numeric)

df_combined.iloc[50553, df_combined.columns.get_loc('rating')] = 1
df_combined.iloc[50552, df_combined.columns.get_loc('rating')] = 1

print("Before group by:")
print(df_combined)

aggregation_functions = {'rating': 'sum', 'movieId': 'first', 'title': 'first'}
df_combined = df_combined.groupby(['userId','genres']).aggregate(aggregation_functions).reset_index()

print("After group by:")
print(df_combined)

ratings_matrix = df_combined.pivot(index='userId', columns='genres', values='rating')
ratings_matrix.replace(np.nan, 0, inplace=True)

print("After pivot:")
print(ratings_matrix)

print(ratings_matrix.iloc[0].values.tolist())

print(getDistanceUsers(0, 1, ratings_matrix))
