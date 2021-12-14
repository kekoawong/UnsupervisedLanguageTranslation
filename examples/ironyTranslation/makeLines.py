from makePickleData import cleaner
from nltk.tokenize import word_tokenize
import csv


if __name__ == "__main__":
    inputFile1 = 'data/redditIrony.csv'
    inputFile2 = 'data/twitterTrain.csv'
    inputFile3 = 'data/twitterTest.csv'
    allWordsFile = 'data/allWords.txt'
    ironicSenFile = 'data/ironicSentences.txt'
    unironicSenFile = 'data/unironicSentences.txt'

    iFile = open(ironicSenFile, 'w')
    uFile = open(unironicSenFile, 'w')
    allFile = open(allWordsFile, 'w')

    with open(inputFile1, newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            # skip header
            next(spamreader, None)
            for row in spamreader:
                # clean string
                line = cleaner(row[0])
                tokens = word_tokenize(line)
                sen = ' '.join(tokens)
                allFile.write(sen + ',' + str(row[1]) + '\n')
                sen += '\n'
                # lowercase, but this may help predict the stylig
                # get rid of all punctuation not periods
                if int(row[1]) < 0:
                    uFile.write(sen)
                else:
                    iFile.write(sen)

    # twitter data, need to convert labels
    with open(inputFile2, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        # skip header
        next(spamreader, None)
        for row in spamreader:
            # skip if not ironic or regular
            if row[1] != 'irony' and row[1] != 'regular':
                continue
            else:
                row[1] = 1 if row[1] == 'irony' else -1
            
            # clean string
            line = cleaner(row[0])
            tokens = word_tokenize(line)
            sen = ' '.join(tokens)
            allFile.write(sen + ',' + str(row[1]) + '\n')
            sen += '\n'

            if int(row[1]) < 0:
                uFile.write(sen)
            else:
                iFile.write(sen)

    # twitter data, need to convert labels
    with open(inputFile3, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        # skip header
        next(spamreader, None)
        for row in spamreader:
            # skip if not ironic or regular
            if row[1] != 'irony' and row[1] != 'regular':
                continue
            else:
                row[1] = 1 if row[1] == 'irony' else -1
            
            # clean string
            line = cleaner(row[0])
            tokens = word_tokenize(line)
            sen = ' '.join(tokens)
            allFile.write(sen + ',' + str(row[1]) + '\n')
            sen += '\n'

            if int(row[1]) < 0:
                uFile.write(sen)
            else:
                iFile.write(sen)
                
    print("Complete!")