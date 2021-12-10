from makePickleData import cleanString, cleaner
from nltk.tokenize import word_tokenize
import csv


if __name__ == "__main__":
    inputFile = '../data/irony-labeled.csv'
    ironicSenFile = '../data/ironicSentences.txt'
    unironicSenFile = '../data/unironicSentences.txt'

    iFile = open(ironicSenFile, 'w')
    uFile = open(unironicSenFile, 'w')

    with open(inputFile, newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            # skip header
            next(spamreader, None)
            for row in spamreader:
                # clean string
                line = cleaner(row[0])
                tokens = word_tokenize(line)
                sen = ' '.join(tokens)
                # lowercase, but this may help predict the stylig
                # get rid of all punctuation not periods
                if int(row[1]) < 0:
                    uFile.write(sen)
                else:
                    iFile.write(sen)
                
    print("Complete!")