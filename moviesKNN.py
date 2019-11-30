import csv
import scipy.spatial
import time

m_genres = []
m_movieDict = {}
m_userDict = {}
m_max_tag_id = 0
m_max_movie_id = 0

m_movies_file_name = 'movies.csv'
m_genome_scores_file_name = 'genome-scores.csv'
m_genome_tags_file_name = 'genome-tags.csv'
m_val_ratings_binary_file_name = 'val_ratings_binary.csv'

# def getTruncatedValRatingsBinary():
#     with open(m_val_ratings_binary_file_name, encoding='utf-8') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter = ',')
#         next(csv_reader, None) # skip the headers
#         min_movie_id_rated = m_max_movie_id
#         for row in csv_reader:
#             movie_id = int(float(row[1]))
#             if (movie_id < min_movie_id_rated):
#               min_movie_id_rated = movie_id
#             if (movie_id <= m_max_movie_id):
#               m_val_ratings_binary_truncated.append(row)
#         print(min_movie_id_rated)
#         print(m_val_ratings_binary_truncated)

def getGenres():
    with open(m_movies_file_name, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        next(csv_reader, None) # skip the headers
        for row in csv_reader:
            for genre in row[2].split('|'):
                if genre not in m_genres:
                    m_genres.append(genre)
        m_genres.sort()
        for genre in m_genres:
            if ("genres" in genre):
                m_genres.remove(genre)
        print("Genres list: " + str(m_genres))

def getMaxTagId():
    with open(m_genome_tags_file_name, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        next(csv_reader, None) # skip the headers
        for row in csv_reader:
            global m_max_tag_id
            m_max_tag_id = int(row[0])
        print("Max tag id: " + str(m_max_tag_id))

def getMaxMovieId():
    with open(m_movies_file_name, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        next(csv_reader, None) # skip the headers
        for row in csv_reader:
            global m_max_movie_id
            m_max_movie_id = int(row[0])
        print("Max movie id: " + str(m_max_movie_id))

def buildMovieInfo():

    # get maximum tag id
    getMaxTagId()

    # get maximum movie id
    getMaxMovieId()

    # populate genres list based movies list
    getGenres()

    # populate genre vector for movie
    with open(m_movies_file_name, encoding = 'utf-8') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter = ',')
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            m_movieDict[row[0]]=[0]*2
            genres_vector = [0]*len(m_genres)
            for genre in row[2].split('|'):
                genres_vector[m_genres.index(genre)] = 1
            m_movieDict[row[0]][0] = genres_vector

    # populate tags vector for movie
    with open(m_genome_scores_file_name, encoding = 'utf-8') as genome_scores:
        genome_scores_reader = csv.reader(genome_scores,delimiter = ',')
        next(genome_scores_reader, None)  # skip the headers
        currentTag = 1
        scores_list = [0] * m_max_tag_id #num tags
        for row in genome_scores_reader:
            scores_list[int(row[1])-1] = float(row[2])
            if (currentTag == m_max_tag_id-1):
                row = next(genome_scores_reader)
                scores_list[int(row[1])-1] = float(row[2])
                m_movieDict[row[0]][1] = scores_list
                scores_list = [0]*m_max_tag_id
                currentTag = 0
            currentTag += 1

def getDistanceMovies(movieA, movieB):
    aGenres = m_movieDict[movieA][0]
    bGenres = m_movieDict[movieB][0]
    aTags = m_movieDict[movieA][1]
    bTags = m_movieDict[movieB][1]
    distance = scipy.spatial.distance.cosine(aGenres,bGenres)
    distance = distance + scipy.spatial.distance.cosine(aTags,bTags)
    return distance

# the high level logic here is to just iterate through
# the val_ratings_binary file and then populate two vectors
# for each user, the first vector represents movies they
# liked, second vector represents movies they disliked.
def populateUsersMovieVectors():
    with open(m_val_ratings_binary_file_name, encoding = 'utf-8') as genome_scores:
        csv_reader = csv.reader(genome_scores,delimiter = ',')
        next(csv_reader, None)  # skip the headers
    for row in csv_reader:
        user_id = int(row[0])
        if user_id not in m_userDict:
            global m_max_movie_id
            m_userDict[user_id] = [0]*m_max_movie_id
        movie_id = int(float(row[1]))
        rating = int(row[2])
        if (rating == 0):
            m_userDict[user_id][movie_id-1] = -1
        else:
            m_userDict[user_id][movie_id-1] = 1
        
    print(m_userDict)

# the main issue I'm running into with this function is that
# cosine distance doesn't work when one of the vectors has only
# zeros.
# not sure of the best way to address that, i guess we can
# either consider distances between users infinity if one
# of their like or dislike vectors is zero, since it means we
# don't have rating data for them?
def getDistanceUsers(userA, userB):
    aMovieRatingsVector = m_userDict[userA]
    bMovieRatingsVector = m_userDict[userB]

    distance = scipy.spatial.distance.cosine(aMovieRatingsVector, bMovieRatingsVector)

    return distance

def getKNNMovies(currentMovie,k):
    with open(m_movies_file_name, encoding = 'utf-8') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter = ',')
        next(csv_reader, None)  # skip the headers
        similarIDs = []
        for row in csv_reader:
            if(row[0]!= currentMovie):
                dist = getDistanceMovies(currentMovie,row[0])
                similarIDs.append((row[0],dist))
        similarIDs.sort(key = lambda x: x[1])
        return similarIDs[:k]

def getKNNUsers(currentUser, k):
    with open(m_val_ratings_binary_file_name, encoding = 'utf-8') as csv_file:
          csv_reader = csv.reader(csv_file, delimiter = ',')
          next(csv_reader, None) # skip the headers
          similarIDs = []
          for row in csv_reader:
              userID = int(row[0])
              if (userID != currentUser):
                 dist = getDistanceUsers(currentUser, userID)
                 similarIDs.append(userID, dist)
          similarIDs.sort(key = lambda x: x[1])
          return similarIDs[:k]

def getCurrentTimeMs():
    return time.time_ns() // 1000000

print("Getting max movie id...")
getMaxMovieId()
print("Populating user movie vectors...")
populateUserMovieVectorsStartTime = getCurrentTimeMs()
populateUsersMovieVectors()
populateUserMovieVectorsFinishTime = getCurrentTimeMs()
print("Finished populating user movie vectors")
print("Getting single user distance...")
getSingleUserDistanceStartTime = getCurrentTimeMs()
getDistanceUsers(54474, 103635)
getSingleUserDistanceFinishTime = getCurrentTimeMs()
print("Finished getting single user distance")
print("Final statistics: " + "\n" +
      "Populate user movie vectors time: " + str(populateUserMovieVectorsFinishTime - populateUserMovieVectorsStartTime) + "\n" +
      "Get single user distance time: " + str(getSingleUserDistanceFinishTime - getSingleUserDistanceStartTime))
# buildMovieInfo()
# print(getKNNMovies('2',1))
# populateUsersMovieVectors()
# print(m_userDict)