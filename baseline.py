import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# get data
dataFile = 'data/allData.pickle'

with open(dataFile, 'rb') as dFile:
    data = pickle.load(dFile)

# separate into training and testing
xTrain, xTest, yTrain, yTest = train_test_split(data, test_size=0.2,random_state=20)

# train and get variables
classifier = nltk.NaiveBayesClassifier.train(trainData)
accuracy = nltk.classify.accuracy(classifier, testData) * 100

print("Classifier accuracy percent:", accuracy)