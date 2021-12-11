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

Run example command:  
python3 unsupervisedTranslation.py --dataf examples/ironyTranslation/data/ironicSentences.txt --datat examples/ironyTranslation/data/unironicSentences.txt --initial examples/ironyTranslation/data/roughUnironicSent.txt -o examples/ironyTranslation/outputs/ironic_to_unironic --savetf examples/ironyTranslation/models/target-forein.torch --saveft examples/ironyTranslation/models/foreign-target.torch