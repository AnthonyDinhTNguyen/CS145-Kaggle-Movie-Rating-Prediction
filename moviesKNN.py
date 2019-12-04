
import csv
import scipy.spatial
genres =['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movieDict = {}
def getGenres():
    with open('../movies.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter =',')
        for row in csv_reader:
            for genre in row[2].split('|'):
                if genre not in genres:
                    genres.append(genre)
        genres.sort()
        for genre in genres:
            if ("genres" in genre):
                genres.remove(genre)
        print(genres)

def buildMovieInfo():
    with open('../movies.csv', encoding = 'utf-8') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter = ',')
        for row in csv_reader:
            movieDict[row[0]]=[0]*2
            tempList = [0]*19
            for genre in row[2].split('|'):
                if("genres" not in genre):
                    tempList[genres.index(genre)] = 1
            movieDict[row[0]][0] = tempList
    with open('../genome-scores.csv', encoding = 'utf-8') as genome_scores:
        genome_scores_reader = csv.reader(genome_scores,delimiter = ',')
        next(genome_scores_reader)
        currentTag = 1
        scores_list = [0]*1128 #num tags
        for row in genome_scores_reader:
            scores_list[int(row[1])-1] = float(row[2])
            if(currentTag == 1127):
                row = next(genome_scores_reader)
                scores_list[int(row[1])-1] = float(row[2])
                movieDict[row[0]][1] = scores_list
                scores_list = [0]*1128
                currentTag=0
            currentTag +=1
                
            
def getDistance(movieA, movieB):
    aGenres = movieDict[movieA][0]
    bGenres = movieDict[movieB][0]
    aTags = movieDict[movieA][1]
    bTags = movieDict[movieB][1]
    distance = scipy.spatial.distance.cosine(aGenres,bGenres)
    distance =distance +scipy.spatial.distance.cosine(aTags,bTags)
    return distance

def getKNNMovies(currentMovie,k):
    buildMovieInfo()
    with open('../movies.csv', encoding = 'utf-8') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter = ',')
        next(csv_reader)#skip the header row
        similarIDs = []
        for row in csv_reader:
            if(row[0]!= currentMovie):
                dist= getDistance(currentMovie,row[0])
                similarIDs.append((row[0],dist))
        similarIDs.sort(key = lambda x: x[1])
        return similarIDs[:k]

print(getKNNMovies('2',30))
#buildMovieInfo()
#print(movieDict['2'][1])
#print(scipy.spatial.distance.cosine([1,1,0],[1,1,0]))
