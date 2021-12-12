

if __name__ == "__main__":
    import argparse, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str, help='training data')
    args = parser.parse_args()

    chineseSent = []
    englishSent = []
    with open('allChinese.txt', 'r') as chinese:
        for line in chinese:
            chineseSent.append(line.rstrip())
    with open('allEnglish.txt', 'r') as english:
        for line in english:
            englishSent.append(line.rstrip())

    if args.outfile:
        with open(args.outfile, 'w') as writeFile:
            for i, eline in enumerate(englishSent):
                outString = f'{chineseSent[i]}\t{eline}\n'
                writeFile.write(outString)
