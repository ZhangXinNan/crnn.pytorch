import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


model_path = './data/crnn.pth'
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)
print("converter.alphabet   :", converter.alphabet)
print("converter.dict       :", converter.dict)
'''
converter.alphabet = '0123456789abcdefghijklmnopqrstuvwxyz-'
converter.dict = {'0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
    'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17,
    'h': 18, 'i': 19, 'j': 20, 'k': 21, 'l': 22, 'm': 23, 'n': 24,
    'o': 25, 'p': 26, 'q': 27, 'r': 28, 's': 29, 't': 30,
    'u': 31, 'v': 32, 'w': 33, 'x': 34, 'y': 35, 'z': 36}
'''
transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
print(img_path, "image.size:", image.size)
# image.size = (184, 72)

image = transformer(image)
print("image.shape:", image.shape)
# image.shape = torch.Size([1, 32, 100])

if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
print("image.shape:", image.shape)
# image.shape = torch.Size([1,1,32,100])
image = Variable(image)
print("image.shape:", image.shape)

model.eval()
preds = model(image)
print("preds.shape:", preds.shape)
# preds.shape: torch.Size([26, 1, 37])

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
# a-----v--a-i-l-a-bb-l-e--- => available
