import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
import re
import pickle
from nltk.tokenize.treebank import TreebankWordDetokenizer

def cleaner(line):
    line = re.sub("@[A-Za-z0-9]+","",line) #Remove @ sign
    line = re.sub(r'\[(.*?)\]\(.+?\)', "", line)
    line = line.replace("\n", "")
    line = line.replace("\r", "")
    line = re.sub(r'\*', "", line) # Remove *
    line = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", line) #Remove http links
    line = re.sub(r'\[(.*?)\]\(.+?\)', "", line)
    line = " ".join(line.split())
    line = ''.join(c for c in line if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    line = line.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text

    # tokenize and retokenize into better format
    tokens = nltk.word_tokenize(str(line))
    line = TreebankWordDetokenizer().detokenize(tokens)
    return line

if __name__ == "__main__":
    # set variables
    percentTraining = 0.7
    inputFile1 = 'data/redditIrony.csv'
    inputFile2 = 'data/twitterTrain.csv'
    inputFile3 = 'data/twitterTest.csv'
    outputFile = 'data/allData.pickle'
    fullSentenceFile = 'data/sentences.pickle'

    # convert stopwords to dict
    sw = {}
    for w in stopwords.words():
        sw[w] = True

    data = []
    sentences = []

    # read through all files 
    with open(inputFile1, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        # skip header
        next(spamreader, None)
        for row in spamreader:
            # clean string
            line = cleaner(row[0])
            sentences.append( (line, row[1]) )
            # tokenize and remove stopwords
            tokens = word_tokenize(line)
            words = [w for w in tokens if not w in sw]
            # append to data
            data.append( (words, row[1]) )
    
    # twitter data, need to convert labels
    with open(inputFile2, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        # skip header
        next(spamreader, None)
        for row in spamreader:
            # clean string
            line = cleaner(row[0])

            # skip if not ironic or regular
            if row[1] != 'irony' and row[1] != 'regular':
                continue
            else:
                row[1] = 1 if row[1] != 'irony' else -1
            
            sentences.append( (line, row[1]) )
            # tokenize and remove stopwords
            tokens = word_tokenize(line)
            words = [w for w in tokens if not w in sw]
            # append to data
            data.append( (words, row[1]) )
    with open(inputFile3, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        # skip header
        next(spamreader, None)
        for row in spamreader:
            # clean string
            line = cleaner(row[0])
            # skip if not ironic or regular
            if row[1] != 'irony' and row[1] != 'regular':
                continue
            else:
                row[1] = 1 if row[1] != 'irony' else -1
            
            sentences.append( (line, row[1]) )
            # tokenize and remove stopwords
            tokens = word_tokenize(line)
            words = [w for w in tokens if not w in sw]
            # append to data
            data.append( (words, row[1]) )

    
    with open(outputFile, 'wb') as outFile:
        pickle.dump(data, outFile)
    with open(fullSentenceFile, 'wb') as senFile:
        pickle.dump(sentences, senFile)

    print("Complete!")