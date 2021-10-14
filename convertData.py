import csv
import nltk.classify.naivebayes as nb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.utils import shuffle
import pickle

# set variables
percentTraining = 0.7
inputFile = 'data/irony-labeled.csv'
trainingFile = 'data/ironyTraining.li'
testingFile = 'data/ironyTesting.li'

# convert stopwords to dict
sw = {}
for w in stopwords.words():
    sw[w] = True

with open(inputFile, newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    data = []
    for row in spamreader:
        # clean string
        line = row[0]
        line = line.replace("\n", "")
        line = line.replace("\r", "")
        # remove links
        line = re.sub(r'\[(.*?)\]\(.+?\)', "", line)
        # tokenize and remove stopwords
        tokens = word_tokenize(line)
        words = [w for w in tokens if not w in sw]
        # append to data
        data.append( (words, row[1]) )
# remove header
data.pop(0)
# shuffle and save data
data = shuffle(data)
splitInt = int( len(data)*percentTraining )
train = open(trainingFile)
pickle.dump(data[:splitInt], train)
test = open(testingFile)
pickle.dump(data[splitInt:], test)