import moviesKNN as mKNN
import usersKNN as uKNN
import pandas as pd
import numpy as np
import csv
movieDict = {}
def KNN():
    with open('submission.csv',mode = 'w') as submissionFile:
        k = 1000
        n= 1000
        csv_writer = csv.writer(submissionFile,delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        testNum = 1
        df_test_cases = pd.read_csv('../test_ratings.csv',sep = ',',names =['userId','movieId'], skiprows =1 )
        i = 0
        userInfo = uKNN.prepareUserInfo()
        mKNN.buildMovieInfo()
        training = pd.read_csv('train_ratings_binary.csv')
        while (i<len(df_test_cases)):
            UID = df_test_cases.iloc[i]
            MID = df_test_cases.iloc[i]
            KNNUsers = uKNN.getKNNUsers(UID,userInfo,k)
            KNNMovies = mKNN.getKNNMovies(MID,n)
            #now get the ratings of how the KNNUsers rated the KNNMovies
            df_filtered = training[training.userId.isin(KNNUsers)] 
            df_filtered = df_filtered[df_filtered.movieId.isin(KNNMovies)]
            if(len(df_filtered)==0):
                print(i)#print the testcase for debugging 
            i+=1