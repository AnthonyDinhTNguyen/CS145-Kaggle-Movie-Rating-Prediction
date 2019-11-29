import csv
import scipy.spatial

m_genres = []
m_movieDict = {}
m_max_tag_id = 0

m_movies_file_name = 'movies-truncated.csv'
m_genome_scores_file_name = 'genome-scores-truncated.csv'
m_genome_tags_file_name = 'genome-tags.csv'

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

def buildMovieInfo():

    # get maximum tag id
    getMaxTagId()

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
                currentTag=0
            currentTag += 1

def getDistance(movieA, movieB):
    aGenres = m_movieDict[movieA][0]
    bGenres = m_movieDict[movieB][0]
    aTags = m_movieDict[movieA][1]
    bTags = m_movieDict[movieB][1]
    distance = scipy.spatial.distance.cosine(aGenres,bGenres)
    distance = distance + scipy.spatial.distance.cosine(aTags,bTags)
    return distance

def getKNNMovies(currentMovie,k):
    with open(m_movies_file_name, encoding = 'utf-8') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter = ',')
        next(csv_reader, None)  # skip the headers
        similarIDs = []
        for row in csv_reader:
            if(row[0]!= currentMovie):
                dist = getDistance(currentMovie,row[0])
                similarIDs.append((row[0],dist))
        similarIDs.sort(key = lambda x: x[1])
        return similarIDs[:k]

buildMovieInfo()
print(getKNNMovies('2',1))
