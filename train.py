#-*-coding:utf8-*-

__author__ = "buyizhiyou"
__date__  = "2017-10-30"



from inception3 import Inception3
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from data_process import MyDataset
from  torchvision import transforms,utils
from torch.utils.data import DataLoader
import  torch
import pdb


train_data = MyDataset(txt='./data/train.txt',transform=transforms.ToTensor())
train_loader = DataLoader(train_data,batch_size=64,shuffle=True) #返回的是迭代器
test_data = MyDataset(txt='./data/val.txt',transform=transforms.ToTensor())
test_loader = DataLoader(test_data,batch_size=64)


model = Inception3()#使用gpu，将模型加载到显存
model = model.cuda()
#print(model)
#print(list(model.parameters()))
loss_func = nn.CrossEntropyLoss()
#开始训练
for epoch in range(10):

	#training------------------------
	train_loss = 0
	train_accu = 0
	step = 0
	for batch_x,batch_y in train_loader:
		batch_x,batch_y = Variable(batch_x).cuda(),Variable(batch_y).cuda()#数据加载到显存
		out = model(batch_x)
		out1 = out[0]

		optimizer = optim.Adam(model.parameters(),lr=0.001)
		#pdb.set_trace()
		loss = loss_func(out1,batch_y)
		train_loss += loss.data[0]
		pred = torch.max(out1,1)[1]
		train_correct = (pred==batch_y).sum()
		train_accu += train_correct.data[0]
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print("EPOCH:%d,STEP:%d"%(epoch,step))
		step = step + 1
	mean_loss = train_loss/(len(train_data))
	mean_accu = train_accu/(len(train_data))
	#print(mean_loss,mean_accu)
	print('Training Loss : %.6f,Accu: %.6f'%(mean_loss,mean_accu))
											
	#evaluation------------------------
	# model.eval()
	# eval_loss = 0
	# eval_accu = 0
	# for batch_x,batch_y in test_loader:
	# 	batch_x,batch_y = Variable(batch_x,volatile=True).cuda(),Variable(batch_y,volatile=True).cuda()
	# 	out = model(batch_x)
	# 	out1 = out[0]
	# 	loss = loss_func(out1,batch_y)
	# 	eval_loss += loss.data[0]
	# 	pred = torch.max(out1,1)[1]
	# 	num_correct = (pred == batch_y).sum()
	# 	eval_accu += num_correct.data[0]
	# mean_loss = eval_loss/(len(test_data))
	# mean_accu = eval_accu/(len(test_data))
	# print('Testing Loss:%.6f,Accu:%.6f'%(mean_loss,mean_accu))

	name = 'inception3'+'_'+str(epoch)+'.pkl'
	torch.save(model,'./models/'+name)