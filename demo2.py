import torch
# from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn2 as crnn


model_path = './data/crnn.pth'
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)   # 37 = len(alphabet) + 1
# 选择设备
if torch.cuda.is_available():
    model = model.cuda()
# 加载模型
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))
# 创建转换器，测试阶段用于将CTC生成的路径转换成最终序列，使用英文字典时忽略大小写
converter = utils.strLabelConverter(alphabet)
# 图像大小转换器
transformer = dataset.resizeNormalize((100, 32))
# 读取图像并转换为100 * 32
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())    # (b,c,h,w) (1,1,32,100)
# image = Variable(image)

model.eval()
preds = model(image)    # (w, c, nclass) (26, 1, 37) 26为ctc生成路径长度也是传入RNN的时间步长，1是batch_size，37是字符类别数

_, preds = preds.max(2)     # 取可能性最大的indices size (26, 1)
preds = preds.transpose(1, 0).contiguous().view(-1)

# preds_size = Variable(torch.IntTensor([preds.size(0)]))
preds_size = torch.IntTensor([preds.size(0)])
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
