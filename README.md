# Unsupervised Machine Translation
This repository contains an implementation of unsupervised machine translation, primarily for the primary purpose of translating text styles without a parallel translation dataset. Implemented using Neural Networks built with [PyTorch](http://pytorch.org/) and natural language processing methods. 

## Dependencies
* [Python3](https://www.python.org)
* [PyTorch](http://pytorch.org/)
* [Sklearn](https://scikit-learn.org/stable/index.html)
* [Pickle](https://docs.python.org/3.8/library/pickle.html)

## Training
Given a dataset of a foreign language sentences, a dataset of target language sentences, and a dataset of an initial rough translation in the target language from the foreign language, the unsupervised model can be trained as follows:
```
usage: unsupervisedTranslation.py [-h] [-f DATAF] [-t DATAT] [--initial INITIAL] [--percentTrain PERCENTTRAIN] [--epochs EPOCHS] [--iterations ITERATIONS] [-o OUTFILE] [--load LOAD][--savetf SAVETF] [--saveft SAVEFT]

optional arguments:
  -h, --help            show this help message and exit
  -f DATAF, --dataf DATAF
                        foreign language data
  -t DATAT, --datat DATAT
                        target language data
  --initial INITIAL     Initial rough translation of target language from foreign language data
  --percentTrain PERCENTTRAIN
                        Percent to be used for training data (in decimal form), the remaining will be split between dev and test
  --epochs EPOCHS, -e EPOCHS
                        Number of epochs to train per model per iteration
  --iterations ITERATIONS, -i ITERATIONS
                        Number of iterations to go through
  -o OUTFILE, --outfile OUTFILE
                        write translations to file
  --load LOAD           load model from file
  --savetf SAVETF       save target to foreign model in file
  --saveft SAVEFT       save foreign to target model in file
```

## Examples
There are two examples data sources included in this repository for the unsupervised training: ironyTranslation and backstrokeSubtitles. The ironyTranslation example is the primary example, used for a Natural Language Processing Project, while backstrokeSubtitles is an example used for experimentation. 

### ironyTranslation
Using a combination of prelabeled datasets from Kaggle, one from [Twitter Tweets](https://www.kaggle.com/nikhiljohnk/tweets-with-sarcasm-and-irony ) and one from [Reddit Comments](https://www.kaggle.com/rtatman/ironic-corpus), the goal of this example was to label ironic statements and then translate these ironic statements into unironic statements.
* [baseline.py](https://github.com/kekoawong/UnsupervisedLanguageTranslation/blob/main/examples/ironyTranslation/baseline.py): Baseline classification of ironic sentences using Naive Bayes classification., scoring 53.12% prediction accuracy.
* [baselineNegation.py](https://github.com/kekoawong/UnsupervisedLanguageTranslation/blob/main/examples/ironyTranslation/baselineNegation.py): Script to roughly negate the sentence, theoretically roughly translating an ironic statement into an unironic statement. 
* [rnnClassifier.py](https://github.com/kekoawong/UnsupervisedLanguageTranslation/blob/main/examples/ironyTranslation/rnnClassifier.py): Classifier built using Recurrent Neural Networks and PyTorch, scoring a 92.3% prediction accuracy.
* [interactive.py](https://github.com/kekoawong/UnsupervisedLanguageTranslation/blob/main/examples/ironyTranslation/interactive.py): Interactive system that takes in a user input, predicting whether the input is ironic and translating the statement into an unironic statement if it is. 
* [data](https://github.com/kekoawong/UnsupervisedLanguageTranslation/blob/main/examples/ironyTranslation/data): Contains the initial data used for the example
* [models](https://github.com/kekoawong/UnsupervisedLanguageTranslation/blob/main/examples/ironyTranslation/models): Saved models from training, foreign-target.torch and target-forein.torch are from the unsupervised translation script while the ModelRnnClassifier-%accuracy.torch are the rnn classification models.
* [outputs](https://github.com/kekoawong/UnsupervisedLanguageTranslation/blob/main/examples/ironyTranslation/outputs): Various outputs from the trained models, comparing the text with the output text or classification from the training.

Example unsupervised training command, from the root directory:
```
python3 unsupervisedTranslation.py --dataf examples/ironyTranslation/data/ironicSentences.txt --datat examples/ironyTranslation/data/unironicSentences.txt --initial examples/ironyTranslation/data/roughUnironicSent.txt -o examples/ironyTranslation/outputs/ironic_to_unironic --savetf examples/ironyTranslation/models/target-forein.torch --saveft examples/ironyTranslation/models/foreign-target.torch --epochs 2 --iterations 4  
```

### backstrokeSubtitles
Using data from subtitle translations from the movie *Star Wars: Episode III – Revenge of the Sith*, used unsupervised approach to experiment with translations.   
* [parse.py](https://github.com/kekoawong/UnsupervisedLanguageTranslation/blob/main/examples/backstrokeSubtitles/parse.py): Parse initial text data.
* [roughEnglish.txt](https://github.com/kekoawong/UnsupervisedLanguageTranslation/blob/main/examples/backstrokeSubtitles/roughEnglish.txt): Initial rough translation into the target language.
* [models](https://github.com/kekoawong/UnsupervisedLanguageTranslation/blob/main/examples/backstrokeSubtitles/models): Saved models from training, foreign-target.torch and target-forein.torch are from the unsupervised translation script.
* [outputs](https://github.com/kekoawong/UnsupervisedLanguageTranslation/blob/main/examples/backstrokeSubtitles/outputs): Various outputs from the trained models, comparing the text with the output text.

Example unsupervised training command, from the root directory:
```
python3 unsupervisedTranslation.py --dataf examples/backstrokeSubtitles/chinese.txt --datat examples/backstrokeSubtitles/english.txt --initial examples/backstrokeSubtitles/roughEnglish.txt -o examples/backstrokeSubtitles/outputs/chinese_to_english --savetf examples/backstrokeSubtitles/models/english-chinese.torch --saveft examples/backstrokeSubtitles/models/chinese-english.torch --epochs 2 --iterations 4 
```
