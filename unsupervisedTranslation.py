import torch
from translationModel import *
from sklearn.model_selection import train_test_split
import random

if __name__ == "__main__":
    import argparse, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataf', type=str, help='foreign language data')
    parser.add_argument('--datat', type=str, help='target language data')
    parser.add_argument('--initial', 'infile', dest='initial', type=str, help='Initial rough translation of foreign language data into target language')
    parser.add_argument('--percentTrain', type=str, help='Percent to be used for training data (in decimal form), the remaining will be split between dev and test')
    parser.add_argument('--epochs', '-e', dest='epochs', type=str, help='Number of epochs to train per model per iteration')
    parser.add_argument('-iterations', '-i', dest='iterations', type=str, help='Number of epochs to train per model per iteration')
    parser.add_argument('-o', '--outfile', dest='iterations', type=str, help='write translations to file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--save', type=str, help='save model in file')
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
        foreignTrain, foreignTest, initialTrain, initialTest, targetTrain, targetTest = train_test_split(dataf, initialTranslation, datat, test_size=1-percentTrain)
        # further split test data into dev and test
        foreignDev, foreignTest, targetDev, targetTest = train_test_split(foreignTest, targetTest, test_size=0.5)

        # Create vocabularies
        fvocab = Vocab()
        tvocab = Vocab()
        initialVocab = Vocab() # this should be same as the 
        for fwords in foreignTrain:
            fvocab |= fwords
        for twords in targetTrain:
            tvocab |= twords
        for roughWords in initialTrain:
            initialVocab |= roughWords

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

        # set variables
        numIterations = 5 if not args.iterations else int(args.iterations)
        numEpochs = 3 if not args.epochs else int(args.epochs)

        # declare optimizers
        opt1 = torch.optim.Adam(target_to_foreign.parameters(), lr=0.0003)
        opt2 = torch.optim.Adam(foreign_to_target.parameters(), lr=0.0003)

        # initialize data
        targetPred = initialTrain

        for iteration in range(numIterations):

            # train target to foreign
            for epoch in range(numEpochs):
                # shuffle data
                traindata = list(zip(targetPred, foreignTrain))
                random.shuffle(traindata)

                ### Update model on train
                train_loss = 0.
                train_fwords = 0
                for twords, fwords in progress(traindata):
                    loss = -m.logprob(twords, fwords)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    train_loss += loss.item()
                    train_fwords += len(fwords) # includes EOS

                ### Validate on dev set and print out a few translations
                dev_loss = 0.
                dev_ewords = 0
                for line_num, (fwords, ewords) in enumerate(devdata):
                    dev_loss -= m.logprob(fwords, ewords).item()
                    dev_ewords += len(ewords) # includes EOS
                    if line_num < 10:
                        translation = m.translate(fwords)
                        print(' '.join(translation))

                if best_dev_loss is None or dev_loss < best_dev_loss:
                    best_model = copy.deepcopy(m)
                    if args.save:
                        torch.save(m, args.save)
                    best_dev_loss = dev_loss

                print(f'[{epoch+1}] train_loss={train_loss} train_ppl={math.exp(train_loss/train_ewords)} dev_ppl={math.exp(dev_loss/dev_ewords)}', flush=True)
                
            m = best_model

        ### Translate test set
        if args.infile:
            with open(args.outfile, 'w') as outfile:
                for fwords in read_mono(args.infile):
                    translation = m.translate(fwords)
                    print(' '.join(translation), file=outfile)