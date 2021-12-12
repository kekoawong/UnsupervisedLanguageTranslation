# nlpProject
Project for Natural Language Processing Course
Unsupervised Machine Translation System without parallel text
Using RNNs. 

## Methods
Using an unsuperived machine learning approach, translate ironic statements to ironic statements  

Works as follows:  
Foreign --> Target  

Use a rough translation system at first (using baselineNegation.py)  
Then, using this rough translation system, initialize two systems  
Pass the output data of each system into the input of the next system.  
System 1: Foreign --> Target  
System 2: Target --> Foreign  
No parallel text.  

Run example command to train irony:  
python3 unsupervisedTranslation.py --dataf examples/ironyTranslation/data/ironicSentences.txt --datat examples/ironyTranslation/data/unironicSentences.txt --initial examples/ironyTranslation/data/roughUnironicSent.txt -o examples/ironyTranslation/outputs/ironic_to_unironic --savetf examples/ironyTranslation/models/target-forein.torch --saveft examples/ironyTranslation/models/foreign-target.torch --epochs 2 --iterations 4  

Run example command to train Backstroke of the West data: 
python3 unsupervisedTranslation.py --dataf examples/backstrokeSubtitles/chinese.txt --datat examples/backstrokeSubtitles/english.txt --initial examples/backstrokeSubtitles/roughEnglish.txt -o examples/backstrokeSubtitles/outputs/chinese_to_english --savetf examples/backstrokeSubtitles/models/english-chinese.torch --saveft examples/backstrokeSubtitles/models/chinese-english.torch --epochs 2 --iterations 4  

python3 attention.py --train examples/backstrokeSubtitles/totalData.txt --dev examples/backstrokeSubtitles/chinese.txt examples/backstrokeSubtitles/chinese.txt -o examples/backstrokeSubtitles/roughEnglish.txt
python3 attention.py --train ../hw2-kekoawong/data/train.zh-en --dev ../hw2-kekoawong/data/dev.zh-en examples/backstrokeSubtitles/chinese.txt -o examples/backstrokeSubtitles/roughEnglish.txt
