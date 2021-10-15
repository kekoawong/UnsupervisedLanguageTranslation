import pickle
import nltk

# get data
dataFile = 'data/sentences.pickle'

with open(dataFile, 'rb') as dFile:
    data = pickle.load(dFile)

lines = []
for line in data:
    sent = line[0]
    for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sent))):
        if (pos.startswith("J")):
            ind = sent.find(word)
            sent1 = sent[:ind]
            sent2 = sent[ind:]
            sent = sent1 + "not " + sent2
            break
        elif (pos.startswith("V")):
            ind = sent.find(word)
            sent1 = sent[:ind]
            sent2 = sent[ind:]
            sent = sent1 + "not " + sent2
            break