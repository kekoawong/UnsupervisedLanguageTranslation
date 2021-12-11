from mmap import MAP_SHARED
import torch
device = 'cpu'

import math, collections.abc, random, copy

from layers import *

def progress(iterable):
    import os, sys
    if os.isatty(sys.stderr.fileno()):
        try:
            import tqdm
            return tqdm.tqdm(iterable)
        except ImportError:
            return iterable
    else:
        return iterable

class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""
    def __init__(self):
        words = {'<EOS>', '<UNK>'}
        self.num_to_word = list(words)    
        self.word_to_num = {word:num for num, word in enumerate(self.num_to_word)}
    def add(self, word):
        if word in self: return
        num = len(self.num_to_word)
        self.num_to_word.append(word)
        self.word_to_num[word] = num
    def discard(self, word):
        raise NotImplementedError()
    def __contains__(self, word):
        return word in self.word_to_num
    def __len__(self):
        return len(self.num_to_word)
    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else: 
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        """Convert a number into a word."""
        return self.num_to_word[num]

def read_parallel(filename):
    """Read data from the file named by 'filename.'

    The file should be in the format:

    我 不 喜 欢 沙 子 \t i do n't like sand

    where \t is a tab character.

    Argument: filename
    Returns: list of pairs of lists of strings. <EOS> is appended to all sentences.
    """
    data = []
    for line in open(filename):
        fline, eline = line.split('\t')
        fwords = fline.split() + ['<EOS>']
        ewords = eline.split() + ['<EOS>']
        data.append((fwords, ewords))
    return data

def read_mono(filename):
    """Read sentences from the file named by 'filename.' 

    Argument: filename
    Returns: list of lists of strings. <EOS> is appended to each sentence.
    """
    data = []
    for line in open(filename):
        words = line.split() + ['<EOS>']
        data.append(words)
    return data
    
class Encoder(torch.nn.Module):
    """IBM Model 2 encoder."""
    
    def __init__(self, vocab_size, dims):
        super().__init__()
        self.emb = Embedding(vocab_size, dims) # This is called V in the notes
        self.rnn = RNN(dims)

    def forward(self, fnums):
        """Encode a Chinese sentence.

        Argument: Chinese sentence (list of n strings)
        Returns: Chinese word encodings (Tensor of size n,d)"""
        input = self.emb(fnums)

        return self.rnn.sequence(input)

class Decoder(torch.nn.Module):
    """IBM Model 2 decoder."""
    
    def __init__(self, dims, vocab_size):
        super().__init__()

        # Layers:
        #   Embedding --> RNN --> Attention --> Tanh --> SoftMax

        # initialize layers
        self.emb = Embedding(vocab_size=vocab_size, output_dims=dims)
        self.rnn = RNN(dims)
        self.att = MaskedSelfAttention(dims)
        self.tanh = TanhLayer(input_dims=2*dims, output_dims=dims)
        self.out = SoftmaxLayer(dims, vocab_size) # This is called U in the notes

    def start(self, fencs):
        """Return the initial state of the decoder.

        Argument:
        - fencs (Tensor of size n,d): Source encodings

        For Model 2, the state is just the English position, but in
        general it could be anything. If you add an RNN to the
        decoder, you should call the RNN's start() method here.

        Returns:
            state in form of 
            (fencs, RNN state)
        """
        
        return (fencs, self.rnn.start())

    def input(self, state, enum):
        """Read in an English word (enum) and compute a new state from
        the old state (state).

        Arguments:
            state: Old state of decoder
            enum:  Next English word (int)

        Returns: New state of decoder
        """
        (fencs, gOld) = state

        u = self.emb.forward(enum)
        gNew = self.rnn.input(gOld, u)
        
        return (fencs, gNew)

    def output(self, state):
        """Compute a probability distribution over the next English word.

        Argument: State of decoder

        Returns: Vector of log-probabilities (tensor of size len(evocab))
        """

        (fencs, gOld) = state
        (recentInput, enc) = gOld
        
        context = attention(enc, fencs, fencs)
        oi = self.tanh.forward(torch.cat((context, enc)))
        return self.out.forward(oi)

class Model(torch.nn.Module):
    """IBM Model 2.

    You are free to modify this class, but you probably don't need to;
    it's probably enough to modify Encoder and Decoder.
    """
    def __init__(self, fvocab, dims, evocab):
        super().__init__()

        # Store the vocabularies inside the Model object
        # so that they get loaded and saved with it.
        self.fvocab = fvocab
        self.evocab = evocab
        
        self.encoder = Encoder(len(fvocab), dims)
        self.decoder = Decoder(dims, len(evocab))

        # This is just so we know what device to create new tensors on        
        self.dummy = torch.nn.Parameter(torch.empty(0))

    def logprob(self, fwords, ewords):
        """Return the log-probability of a sentence pair.

        Arguments:
            fwords: source sentence (list of str)
            ewords: target sentence (list of str)

        Return:
            log-probability of ewords given fwords (scalar)"""

        fnums = torch.tensor([self.fvocab.numberize(f) for f in fwords], device=self.dummy.device)
        fencs = self.encoder(fnums)
        state = self.decoder.start(fencs)
        logprob = 0.
        for eword in ewords:
            o = self.decoder.output(state)
            enum = self.evocab.numberize(eword)
            logprob += o[enum]
            state = self.decoder.input(state, enum)
        return logprob

    def translate(self, fwords):
        """Translate a sentence using greedy search.

        Arguments:
            fwords: source sentence (list of str)

        Return:
            ewords: target sentence (list of str)
        """
        
        fnums = torch.tensor([self.fvocab.numberize(f) for f in fwords], device=self.dummy.device)
        fencs = self.encoder(fnums)
        state = self.decoder.start(fencs)
        ewords = []
        for i in range(100):
            o = self.decoder.output(state)
            enum = torch.argmax(o).item()
            eword = self.evocab.denumberize(enum)
            if eword == '<EOS>': break
            ewords.append(eword)
            state = self.decoder.input(state, enum)
        return ewords

if __name__ == "__main__":
    import argparse, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='training data')
    parser.add_argument('--dev', type=str, help='development data')
    parser.add_argument('infile', nargs='?', type=str, help='test data to translate')
    parser.add_argument('-o', '--outfile', type=str, help='write translations to file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--save', type=str, help='save model in file')
    args = parser.parse_args()

    if args.train:
        # Read training data and create vocabularies
        traindata = read_parallel(args.train)

        fvocab = Vocab()
        evocab = Vocab()
        for fwords, ewords in traindata:
            fvocab |= fwords
            evocab |= ewords

        # Create model
        m = Model(fvocab, 64, evocab) # try increasing 64 to 128 or 256
        
        if args.dev is None:
            print('error: --dev is required', file=sys.stderr)
            sys.exit()
        devdata = read_parallel(args.dev)
            
    elif args.load:
        if args.save:
            print('error: --save can only be used with --train', file=sys.stderr)
            sys.exit()
        if args.dev:
            print('error: --dev can only be used with --train', file=sys.stderr)
            sys.exit()
        m = torch.load(args.load)

    else:
        print('error: either --train or --load is required', file=sys.stderr)
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
