import nltk.classify.naivebayes as nb
import nltk.classify as classify
import pickle

trainingFile = 'data/ironyTraining.li'
testingFile = 'data/ironyTesting.li'

with open(trainingFile) as train:
    trainData = pickle.load(train)
with open(testingFile) as test:
    testData = pickle.load(test)

# train and get variables
classifier = nb.train(trainData)
accuracy = classify.accuracy(classifier, testing_set) * 100

print("Classifier accuracy percent:", accuracy)