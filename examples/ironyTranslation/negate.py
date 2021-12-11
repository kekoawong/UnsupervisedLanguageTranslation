from baselineNegation import negateSentence

if __name__ == "__main__":
    import argparse, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='file to negate sentences')
    args = parser.parse_args()

    if args.infile:
        with open(args.infile, 'r') as infile:
            for line in infile:
                negation = negateSentence(line)
                print(negation)
