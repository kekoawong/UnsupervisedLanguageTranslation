import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

def custom_tokenizer(stringpassed):
    lemmatizer = WordNetLemmatizer()
    lematize_words = [lemmatizer.lemmatize(word) for word in stringpassed]
    return lematize_words

# get data
dataFile = 'data/allData.pickle'

with open(dataFile, 'rb') as dFile:
    data = pickle.load(dFile)

lines = []
for line in data:
    listToStr = ' '.join(map(str, line[0]))
    lines.append(listToStr)
labels = [line[1] for line in data]

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# tfidf scores
vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
tfidf = vectorizer.fit_transform(lines)
# separate into training and testing
xTrain, xTest, yTrain, yTest = train_test_split(tfidf, labels, test_size=0.2, random_state=20)

#Calling the Class
naive_bayes = GaussianNB()
 
#Fitting the data to the classifier
naive_bayes.fit(xTrain.toarray() , yTrain)
 
#Predict on test data
y_predicted = naive_bayes.predict(xTest.toarray())
accuracy = metrics.accuracy_score(y_predicted , yTest)


print("Classifier accuracy percent:", accuracy)