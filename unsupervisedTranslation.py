import torch
from attention import *
from sklearn.model_selection import train_test_split
import random
import time

# declare variables for timing
# declare variables for timing
totalLinesLeft = 0
timePerLine = 0
timeLeft = timePerLine*totalLinesLeft
totalstarttime = time.time()
epochstartTime = time.time()

def trainModel(m, opt, inputData, targetData):
    '''
    Input of Model, optimizer, inputData, and targetData
    Will output Model and optimizer
    Ret: (model, optimizer)
    '''
    # shuffle data
    traindata = list(zip(inputData, targetData))
    totalLen = len(inputData)
    random.shuffle(traindata)

    ### Update model on train
    train_loss = 0.
    train_target_words = 0
    for i, (input_words, target_words) in enumerate(progress(traindata)):
        loss = -m.logprob(input_words, target_words)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()
        train_target_words += len(target_words) # includes EOS
        if i % 100 == 0 and i != 0:
            print(f'        On line {i}/{totalLen}')
            print(len(target_words))
            avgTime = (time.time() - epochstartTime)/i
            timeLeftEpoch = avgTime * (totalLen-i)
            print(f'        Time left for epoch: {round(timeLeftEpoch/60, 0)} mins')

    print(f'        train_loss={train_loss} train_ppl={math.exp(train_loss/train_target_words)}', flush=True)
    return m, opt

def validateDev(m, inputData, targetData):
    ### Validate on dev set and print out a few translations
    devdata = list(zip(inputData, targetData))
    dev_loss = 0.
    dev_ewords = 0
    for line_num, (fwords, twords) in enumerate(devdata):
        dev_loss -= m.logprob(fwords, twords).item()
        dev_ewords += len(twords) # includes EOS
        if line_num < 10:
            translation = m.translate(fwords)
            print(' '.join(translation))

    print(f'        dev_ppl={math.exp(dev_loss/dev_ewords)}', flush=True)
    return dev_loss

def outputPred(m, inputData):
    outputPred = []
    for words in inputData:
        translation = m.translate(words)
        outputPred.append(translation)
    return outputPred


def outputTest(m, fileName, inputData, predType):
    '''
    Writes outputs to file
    predType is the type contained in the file, either target or foreign
    '''
    fileN = f'{fileName}-{predType}-{dev_loss}'
    with open(fileN, 'w') as outfile:
        for fwords in inputData:
            translation = m.translate(fwords)
            initialSent = ' '.join(fwords)
            translatedSent = ' '.join(translation)
            print(f'{initialSent}    {translatedSent}', file=outfile)

    return

if __name__ == "__main__":
    import argparse, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--dataf', dest='dataf', type=str, help='foreign language data')
    parser.add_argument('-t', '--datat', dest='datat', type=str, help='target language data')
    parser.add_argument('--initial', dest='initial', type=str, help='Initial rough translation of target language from foreign language data')
    parser.add_argument('--percentTrain', type=str, help='Percent to be used for training data (in decimal form), the remaining will be split between dev and test')
    parser.add_argument('--epochs', '-e', dest='epochs', type=str, help='Number of epochs to train per model per iteration')
    parser.add_argument('-iterations', '-i', dest='iterations', type=str, help='Number of epochs to train per model per iteration')
    parser.add_argument('-o', '--outfile', dest='outfile', type=str, help='write translations to file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--savetf', dest='savetf', type=str, help='save target to foreign model in file')
    parser.add_argument('--saveft', dest='saveft', type=str, help='save foreign to target model in file')
    args = parser.parse_args()

    if args.dataf and args.initial and args.datat:
        '''
        dataf: Foreign Language data
        datat: Target language data
        initialTranslation: The rough initial translation of the foreign language into the target
        '''

        # Read in data
        dataf = read_mono(args.dataf)
        datat = read_mono(args.dataf)
        initialTranslation = read_mono(args.initial)

        # split training and testing data
        percentTrain = 0.9 if not args.percentTrain else float(args.percentTrain)
        foreignTrain, foreignTest, targetTrain, targetTest = train_test_split(dataf, datat, test_size=1-percentTrain)
        # further split test data into dev and test
        foreignDev, foreignTest, targetDev, targetTest = train_test_split(foreignTest, targetTest, test_size=0.5)

        # Create vocabularies
        fvocab = Vocab()
        tvocab = Vocab()
        for fwords in foreignTrain:
            fvocab |= fwords
        for twords in targetTrain:
            tvocab |= twords
        # for roughWords in initialTrain:
        #     initialVocab |= roughWords

        # Create initial translation models
        # Do we need to update vocabs?
        target_to_foreign = Model(tvocab, 64, fvocab)
        foreign_to_target = Model(fvocab, 64, tvocab) # try increasing 64 to 128 or 256

    else:
        print('error: foreign data, target data, and rough initial translation all required', file=sys.stderr)
        sys.exit()

    if args.initial and not args.outfile:
        print('error: -o is required', file=sys.stderr)
        sys.exit()

    # start training
    if args.dataf and args.initial and args.datat:

        print("Starting to train")
        # set variables
        numIterations = 5 if not args.iterations else int(args.iterations)
        numEpochs = 3 if not args.epochs else int(args.epochs)

        # declare optimizers
        opt_tf = torch.optim.Adam(target_to_foreign.parameters(), lr=0.0003)
        opt_ft = torch.optim.Adam(foreign_to_target.parameters(), lr=0.0003)

        # initialize data
        targetPred = initialTranslation

        best_dev_loss1 = None
        best_dev_loss2 = None

        for iteration in range(numIterations):

            # train target to foreign
            print(f'Iteration {iteration+1}/{numIterations}, Target to Foreign:')
            # set data
            predTrain, predTest = train_test_split(targetPred, test_size=1-percentTrain)
            predDev, predTest = train_test_split(predTest, test_size=0.5)

            for epoch in range(numEpochs):
                epochstartTime = time.time()
                print(f'    Epoch {epoch+1}/{numEpochs}:')
                # train model
                target_to_foreign, opt_tf = trainModel(target_to_foreign, opt_tf, predTrain, foreignTrain)

                # validate dev
                dev_loss = validateDev(target_to_foreign, predDev, targetDev)
                if best_dev_loss1 is None or dev_loss < best_dev_loss1:
                    best_model_tf = copy.deepcopy(target_to_foreign)
                    if args.savetf:
                        torch.save(target_to_foreign, args.savetf)

                    ### Translate test set if good dev scoring
                    if args.outfile:
                        outputTest(target_to_foreign, args.outfile, predTest, 'foreign')

                    best_dev_loss1 = dev_loss
            # update model
            target_to_foreign = best_model_tf

            # train foreign to target
            print(f'Iteration {iteration+1}/{numIterations}, Foreign to Target:')
            # set data
            foreignPred = outputPred(target_to_foreign, targetPred)
            predTrain, predTest = train_test_split(foreignPred, test_size=1-percentTrain)
            predDev, predTest = train_test_split(predTest, test_size=0.5)

            for epoch in range(numEpochs):
                print(f'    Epoch {epoch+1}/{numEpochs}:')
                # train model
                foreign_to_target, opt_ft = trainModel(foreign_to_target, opt_ft, predTrain, foreignTrain)

                # validate dev
                dev_loss = validateDev(foreign_to_target, predDev, targetDev)
                if best_dev_loss2 is None or dev_loss < best_dev_loss2:
                    best_model_ft = copy.deepcopy(foreign_to_target)
                    if args.saveft:
                        torch.save(foreign_to_target, args.saveft)

                    ### Translate test set if good dev scoring
                    if args.outfile:
                        outputTest(foreign_to_target, args.outfile, predTest, 'target')

                    best_dev_loss2 = dev_loss
            # update model
            foreign_to_target = best_model_ft

            # update next iteration data
            targetPred = outputPred(foreign_to_target, foreignPred)
