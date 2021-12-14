#!/usr/bin/env python3
import layers
import torch
import torch.nn as nn
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import time
# declare variables for timing
totalLinesLeft = 0
timePerLine = 0
timeLeft = timePerLine*totalLinesLeft
totalstarttime = time.time()
epochstartTime = time.time()

def create_mapping(vocab, words):
    r = []
    for word in words:
        if word in vocab:
            r.append(vocab.index(word))
        else:
            r.append(vocab.index('<UNK>'))
    return torch.tensor(r)

def write_to_file(filename, sentences, labels):
    with open(filename, 'w') as writeFile:
        for i, sen in enumerate(sentences):
            outputString=" "
            outputString = outputString.join(sen)
            outputString += f'  {labels[i]}\n'
            writeFile.write(outputString)
    
def computeScore(pred, actual):
    # takes in the letters as arguments, or int of indices
    correct = 0
    total = 0
    for i, p in enumerate(pred):
        total += 1
        if p == actual[i]:
            correct += 1
    return correct/total

def create_data(infile, vocab):
    '''
    Takes the input file and vocab as arguments
    Outputs a data object of the list of lists of words and labels
        (list of lists of words, list of lists of labels,
    Should be used for dev and test data
    '''
    # create output list of lists
    sentWords = []
    sentLabels = []
    input = open(infile, 'rb')
    data = pickle.load(input)
    for line in data:
        words = line[0]
        label = line[1]

        # check for words to convert to <UNK>
        for i, word in enumerate(words):
            if word not in vocab:
                words[i] = '<UNK>'
        
        # append to list
        sentWords.append(words)
        sentLabels.append(label)

    return sentWords, sentLabels

def create_all_data(infile):
    '''
    Takes the input file arguments
    Outputs a data object of the list of lists of words and labels:
        (list of lists of words, list of lists of labels, list of vocab, list of labels)
    Should be used for train data
    '''

    print("Creating all data")
    # create vocab and labels
    labels = {}
    vocab = {}

    print("Loading pickle file")
    with open(infile, 'rb') as stuff:
        data = pickle.load(stuff)
    print("Done loading pickle file")
    for line in data:
        words = line[0]
        label = line[1]

        # update vocab and label count
        if label not in labels:
            labels[label] = 0
        labels[label] += 1
        # update vocab
        for word in words:
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1

    # replace single words with <UNK>, updating vocab
    vocab['<UNK>'] = 0
    vocabList = list(vocab.keys())
    for word, num in vocab.items():
        if num == 1:
            vocab['<UNK>'] += 1
            vocabList.remove(word)
    
    sentWords, sentLabels = create_data(infile, vocabList)

    return sentWords, sentLabels, vocabList, list(labels.keys())

# declare nn model
class Model(nn.Module):
    def __init__(self, rnnDim, outputDim, vocab, labels):
        super().__init__()

        # input dims will be vocab length

        # Store the vocabularies inside the Model object
        self.vocab = vocab
        self.vocabLength = len(vocab)
        self.labels = labels

        # initialize layers
        self.embedding = layers.Embedding(self.vocabLength, rnnDim)
        self.rnn1 = layers.RNN(rnnDim)
        self.softmax = layers.SoftmaxLayer(rnnDim, outputDim)
       

    def forward(self, X):
        senL = len(X)
        pred = create_mapping(self.vocab, X)
        #print(f'after mapping: {pred.shape}')
        pred = self.embedding(pred)
        #print(f'after embedding: {pred.shape}')
        pred = self.rnn1.sequence(pred)
        #print(f'after rnn2: {pred.shape}')
        pred = self.softmax.forward(pred)
        #print(f'after sofrmax: {pred.shape}')

        return(pred[senL-1])

    def loss_fn(self, predTensor, label):
        return predTensor[self.labels.index(label)]


if __name__=="__main__":
    # file names
    trainFile = 'data/allData.pickle'

    allData, allLabels, vocab, labels = create_all_data(trainFile)
    trainData, testData, trainLabels, testLabels = train_test_split(allData, allLabels, test_size=0.1)
    print(f'train data: {len(trainData)} trainlabels: {len(trainLabels)}')
    testData, testLabels, devData, devLabels = train_test_split(testData, testLabels, test_size=0.5)

    # Define Model
    numLabels = len(labels)
    m = Model(200, numLabels, vocab, labels)
    # Define optimizer
    opt = torch.optim.Adam(m.parameters(), lr=.0002)

    for epoch in range(10):
        epochstartTime = time.time()
        ### Update model on train
        train_loss = 0

        x, y = shuffle(trainData, trainLabels)
        totalLen = len(x)
        print(f'Entering epoch {epoch}')
        
        for li, line in enumerate(x):
            epochstartTime = time.time()
            # actual = torch.zeros(numLabels)
            # actual[labels.index(y[li])] = 1

            opt.zero_grad()
            pred = m(line)
            loss = -m.loss_fn(pred, y[li])
            loss.backward()
            opt.step()

            if li % 100 == 0 and li != 0:
                # print(f'Tree Score: {tree_score}')
                # print(f'z: {z}')
                avgTime = (time.time() - epochstartTime)/li
                timeLeftEpoch = avgTime * (totalLen-li)
                print(f'        On line {li}/{totalLen}. Time left for epoch: {round(timeLeftEpoch/60, 3)} mins')
                print(f'Train loss on round {li} of {totalLen}: {loss}')

            train_loss += loss

        print(f'Train loss: {train_loss}')

        # label dev data
        devPredL = []
        for li, line in enumerate(devData):
            devPred = m(line)
            la = torch.argmax(devPred)
            devPredL.append(labels[la])

        # compute accuracy
        score = computeScore(devPredL, devLabels)
        print(f'Dev accuracy: {score}')
        write_to_file(f'outputs/devRnnAccuracy{round(score,3)}', devData, devPredL)

        # test data
        testPredL = []
        for li, line in enumerate(testData):
            testPred = m(line)
            la = torch.argmax(testPred)
            testPredL.append(labels[la])

        # compute f1
        score = computeScore(testPredL, testLabels)
        print(f'Test accuracy: {score}')
        write_to_file(f'outputs/testRnnAccuracy{round(score,3)}', testData, testPredL)

        # save model
        filename = f'models/ModelRnnClassifier-{round(score,3)}.torch'
        torch.save(m, filename)