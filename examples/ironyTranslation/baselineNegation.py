import pickle
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer

def negateSentence(sent):
    tokens = nltk.word_tokenize(str(sent))
    for word,pos in nltk.pos_tag(tokens):
        if (pos.startswith("J")):
            ind = tokens.index(word)
            tokens.insert(ind, "not")
            sent = TreebankWordDetokenizer().detokenize(tokens)
            break
        elif (pos.startswith("V")):
            ind = tokens.index(word) + 1
            tokens.insert(ind, "not")
            sent = TreebankWordDetokenizer().detokenize(tokens)
            break
        else:
            tokens.insert(0, "not")
            sent = TreebankWordDetokenizer().detokenize(tokens)
            break
    return sent

if __name__ == "__main__":
    # get data
    dataFile = 'data/sentences.pickle'

    with open(dataFile, 'rb') as dFile:
        data = pickle.load(dFile)

    lines = []
    for line in data:
        sent = negateSentence(line[0])
        
        if line[0] == sent:
            print(f'Not able to translate: {sent}')
            print()
        else:
            if int(line[1]) == 1:
                print('IRONIC --> UNIRONIC')
                print(f'{line[0]}')
                print(f'{sent}')
                print()
            elif int(line[1]) == -1:
                print('UNIRONIC --> IRONIC')
                print(f'{line[0]}')
                print(f'{sent}')
                print()