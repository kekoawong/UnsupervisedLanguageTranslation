import torch
from translationModel import *
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    import argparse, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataf', type=str, help='foreign language data')
    parser.add_argument('--datat', type=str, help='target language data')
    parser.add_argument('--initial', 'infile', dest='initial', type=str, help='Initial rough translation of foreign language data into target language')
    parser.add_argument('--percentTrain', type=str, help='Percent to be used for training data, the remaining will be test')
    parser.add_argument('--epochs', '-e', dest='epochs', type=str, help='Number of epochs to train per model per iteration')
    parser.add_argument('-iterations', '-i', dest='iterations', type=str, help='Number of epochs to train per model per iteration')
    parser.add_argument('-o', '--outfile', dest='iterations', type=str, help='write translations to file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--save', type=str, help='save model in file')
    args = parser.parse_args()

    if args.dataf and args.initial and args.datat:

        # Read in data
        dataf = read_mono(args.dataf)
        datat = read_mono(args.dataf)
        initialTranslation = read_mono(args.initial)

        # split training and testing data
        percentTrain = 0.95 if not args.percentTrain else float(args.percentTrain)
        

        fvocab = Vocab()
        tvocab = Vocab()
        initialTransVocab = Vocab()
        for fwords in dataf:
            fvocab |= fwords
        for twords in datat:
            tvocab |= twords
        for roughWords in initialTranslation:
            initialTransVocab |= roughWords

        # Create initial translation model
        target_to_foreign = Model(fvocab, 64, tvocab) # try increasing 64 to 128 or 256

    else:
        print('error: foreign data, target data, and rough initial translation all required', file=sys.stderr)
        sys.exit()

    if args.infile and not args.outfile:
        print('error: -o is required', file=sys.stderr)
        sys.exit()

    if args.train:
        opt = torch.optim.Adam(m.parameters(), lr=0.0003)

        best_dev_loss = None
        for epoch in range(10):
            random.shuffle(traindata)

            ### Update model on train
            train_loss = 0.
            train_ewords = 0
            for fwords, ewords in progress(traindata):
                loss = -m.logprob(fwords, ewords)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
                train_ewords += len(ewords) # includes EOS

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