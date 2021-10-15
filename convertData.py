import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pickle

# set variables
percentTraining = 0.7
inputFile = 'data/irony-labeled.csv'
outputFile = 'data/allData.pickle'
fullSentenceFile = 'data/sentences.pickle'

# convert stopwords to dict
sw = {}
for w in stopwords.words():
    sw[w] = True

with open(inputFile, newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    data = []
    sentences = []
    for row in spamreader:
        # clean string
        line = row[0]
        line = line.replace("\n", "")
        line = line.replace("\r", "")
        # remove links
        line = re.sub(r'\[(.*?)\]\(.+?\)', "", line)
        sentences.append( (line, row[1]) )
        # tokenize and remove stopwords
        tokens = word_tokenize(line)
        words = [w for w in tokens if not w in sw]
        # append to data
        data.append( (words, row[1]) )
# remove header
data.pop(0)
with open(outputFile, 'wb') as outFile:
    pickle.dump(data, outFile)
with open(fullSentenceFile, 'wb') as senFile:
    pickle.dump(sentences, senFile)

print("Complete!")