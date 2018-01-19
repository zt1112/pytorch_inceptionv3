#-*-coding:utf8-*-

__author__ = "buyizhiyou"
__date__ = "2017-10-18"

import torch
import numpy as np
from torch.autograd import Variable
from data_process import default_loader
import pdb


# classes = ['bus','dinosaur','elephant','flower','horse']
#Predict--------------------------
model = torch.load('./models/inception3_0.pkl').cuda()
im = default_loader('./data/test/0_0.png')
im = np.expand_dims(im,0)#扩展一维，（Ｎ，Ｃ，Ｈ，Ｗ）
im = im.transpose(0,3,1,2)#nhwc===>nchw
im = torch.from_numpy(im).float()
x = Variable(im).cuda()
pred = model(x)
pdb.set_trace()
index = torch.max(pred[0],1)[1].data[0]
print('预测结果:%d'%(index))

