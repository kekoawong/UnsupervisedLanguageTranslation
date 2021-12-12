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
        # skip if empty
        if len(input_words) == 0 or len(target_words) == 0:
            continue
        loss = -m.logprob(input_words, target_words)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()
        train_target_words += len(target_words) # includes EOS
        if i % 100 == 0 and i != 0:
            avgTime = (time.time() - epochstartTime)/i
            timeLeftEpoch = avgTime * (totalLen-i)
            print(f'        On line {i}/{totalLen}. Time left for epoch: {round(timeLeftEpoch/60, 3)} mins')

    print(f'        train_loss={train_loss} train_ppl={math.exp(train_loss/train_target_words)}', flush=True)
    return m, opt

def validateDev(m, inputData, targetData):
    ### Validate on dev set and print out a few translations
    devdata = list(zip(inputData, targetData))
    dev_loss = 0.
    dev_ewords = 0
    for line_num, (fwords, twords) in enumerate(devdata):
        # skip if empty
        if len(fwords) == 0 or len(twords) == 0:
            continue
        dev_loss -= m.logprob(fwords, twords).item()
        dev_ewords += len(twords) # includes EOS
        if line_num < 10:
            translation = m.translate(fwords)
            print(' '.join(translation))

    print(f'        dev_ppl={math.exp(dev_loss/dev_ewords)}', flush=True)
    print(f'Input words: {fwords} --> Output words: {translation}')
    return dev_loss

def outputPred(m, inputData):
    outputPred = []
    for words in inputData:
        try: 
            translation = m.translate(words)
        except TypeError:
            print(f'Bad words: {words}')
            translation = []
        outputPred.append(translation)
    return outputPred


def outputTest(m, fileName, inputData, predType, dev_loss):
    '''
    Writes outputs to file
    predType is the type contained in the file, either target or foreign
    '''
    fileN = f'{fileName}-{predType}-{round(dev_loss, 3)}'
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
        datat = read_mono(args.datat)
        initialTranslation = read_mono(args.initial)

        # temporary
        # num = 500
        # dataf = dataf[:num]
        # datat = datat[:num]
        # initialTranslation = initialTranslation[:num]

        # Create vocabularies
        fvocab = Vocab()
        tvocab = Vocab()
        for fwords in dataf:
            fvocab |= fwords
        for twords in datat:
            tvocab |= twords

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
        percentTrain = 0.9 if not args.percentTrain else float(args.percentTrain)

        best_dev_loss1 = None
        best_dev_loss2 = None

        for iteration in range(numIterations):

            # train target to foreign
            print(f'Iteration {iteration+1}/{numIterations}, Target to Foreign:')
            # set data, need the lines in all the data to correspond
            predTargetTrain, predTargetTest, foreignTrain, foreignTest, targetTrain, targetTest  = train_test_split(targetPred, dataf, datat, test_size=1-percentTrain)
            predTargetDev, predTargetTest, foreignDev, foreignTest, targetDev, targetTest = train_test_split(predTargetTest, foreignTest, targetTest, test_size=0.5)

            for epoch in range(numEpochs):
                epochstartTime = time.time()
                print(f'    Epoch {epoch+1}/{numEpochs}:')
                # train model, should be pred training toward the actual foreign language
                target_to_foreign, opt_tf = trainModel(target_to_foreign, opt_tf, predTargetTrain, foreignTrain)

                # validate dev, should be the dif type of words, so target --> foreign
                dev_loss = validateDev(target_to_foreign, predTargetDev, foreignDev)
                if best_dev_loss1 is None or dev_loss < best_dev_loss1:
                    best_model_tf = copy.deepcopy(target_to_foreign)
                    if args.savetf:
                        torch.save(target_to_foreign, args.savetf)

                    ### Translate test set if good dev scoring
                    if args.outfile:
                        outputTest(target_to_foreign, args.outfile, predTargetTest, 'foreign', dev_loss)

                    best_dev_loss1 = dev_loss
            # update model
            target_to_foreign = best_model_tf

            # train foreign to target
            print(f'Iteration {iteration+1}/{numIterations}, Foreign to Target:')
            # set data
            foreignPred = outputPred(target_to_foreign, targetPred)
            predForeignTrain, predForeignTest, foreignTrain, foreignTest, targetTrain, targetTest  = train_test_split(foreignPred, dataf, datat, test_size=1-percentTrain)
            predForeignDev, predForeignTest, foreignDev, foreignTest, targetDev, targetTest = train_test_split(predForeignTest, foreignTest, targetTest, test_size=0.5)

            for epoch in range(numEpochs):
                print(f'    Epoch {epoch+1}/{numEpochs}:')
                epochstartTime = time.time()
                # train model, should be pred training toward the actual target language
                foreign_to_target, opt_ft = trainModel(foreign_to_target, opt_ft, predForeignTrain, targetTrain)

                # validate dev,  should be the dif type of words, so pred foreign --> actual target
                dev_loss = validateDev(foreign_to_target, predForeignDev, targetDev)
                if best_dev_loss2 is None or dev_loss < best_dev_loss2:
                    best_model_ft = copy.deepcopy(foreign_to_target)
                    if args.saveft:
                        torch.save(foreign_to_target, args.saveft)

                    ### Translate test set if good dev scoring
                    if args.outfile:
                        outputTest(foreign_to_target, args.outfile, predForeignTest, 'target', dev_loss)

                    best_dev_loss2 = dev_loss
            # update model
            foreign_to_target = best_model_ft

            # update next iteration data
            targetPred = outputPred(foreign_to_target, foreignPred)

            print(f'Time left to complete: { (time.time() - totalstarttime) * (numIterations - 1 - iteration) }')
