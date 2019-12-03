
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time

def cur_time_millis():
    return time.time()*1000.0

def print_t(msg):
    print(msg + " (time " + str(cur_time_millis()) + ")")

print_t("Reading in val_ratings_binary.csv...")

df = pd.read_csv('../val_ratings_binary.csv', sep=',', names=['userId', 'movieId', 'rating'])

print_t("Done reading in val_ratings_binary.csv.")

print(df.head())

print(df.describe())

ratings_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')

print(ratings_matrix.head())
