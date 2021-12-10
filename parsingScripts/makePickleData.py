import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
import re
import pickle

def cleaner(line):
    line = re.sub("@[A-Za-z0-9]+","",line) #Remove @ sign
    line = re.sub(r'\[(.*?)\]\(.+?\)', "", line)
    line = re.sub(r'\*', "", line) # Remove *
    line = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", line) #Remove http links
    line = " ".join(line.split())
    line = ''.join(c for c in line if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    line = line.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    #line = " ".join(w for w in nltk.wordpunct_tokenize(line))
    return line

def cleanString(line):
    '''
    Function will take string as input, cleaning links and getting rid of stopwords
    '''
    line = line.replace("\n", "")
    line = line.replace("\r", "")
    # remove links
    line = re.sub(r'\[(.*?)\]\(.+?\)', "", line)
    return line

if __name__ == "__main__":
    # set variables
    percentTraining = 0.7
    inputFile = '../data/irony-labeled.csv'
    outputFile = '../data/allData.pickle'
    fullSentenceFile = '../data/sentences.pickle'

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
            line = cleanString(row[0])
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