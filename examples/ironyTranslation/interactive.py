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
    pred = classifyModel(words)
    i = torch.argmax(pred)
    p = classifyModel.labels[i]
    is_ironic = True if p == 1 else False

    if is_ironic:
        print('Sentence is ironic')
        trans = translationModel.translate(words)
        sent = ' '.join(trans)
        print(f'Translation to unironic: {sent}')
    else:
        print("Sentence is not ironic")


print('Goodbye!')