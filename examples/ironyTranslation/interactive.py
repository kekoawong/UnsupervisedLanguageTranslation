import torch

print('Welcome to the interactive input system!')
cont = True
# initialize models
classifyModelPath = 'models/ModelRnnClassifier-0.926.torch'
translationModelPath = 'models/foreign-target.torch'

from rnnClassifier import Model
classifyModel = torch.load(classifyModelPath)
from attention import Model
translationModel = torch.load(translationModelPath)

while cont:
    inp = input('Enter Input Sentence: ')

    # Get input, split
    words = str(inp).split()

    # Generate classification prediciton
    classification = classifyModel(words)
    
    print(classification)


print('Goodbye!')