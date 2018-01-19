#-*-coding:utf8-*-

__author__ = "buyizhiyou"
__date__ = "2017-10-30"

from torch.utils.data import Dataset,DataLoader
import numpy as np
from PIL import Image
from  torchvision import transforms,utils
import pdb

#数据预处理和加载
def default_loader(path):
	im = Image.open(path).convert('RGB')
	im = np.asarray(im.resize((299,299)))
	#im = im.transpose((2,0,1))
	#print(im.shape)
	return im

class MyDataset(Dataset):
	def __init__(self,txt,transform=None,target_transform=None,loader=default_loader):
		f = open(txt,'r')
		self.folder = txt.split('/')[-1].split('.')[0]
		imgs = []
		for line in f.readlines():
			img_name = line.split()[0]
			label = line.split()[1]
			imgs.append((img_name,int(label)))
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self,index):#类的特殊方法
	
		img_name,label = self.imgs[index]
		img_path = './data/'+self.folder+'/'+img_name
		img = self.loader(img_path)
		if self.transform is not None:
			img = self.transform(img)
		return img,label

	def __len__(self):
		return len(self.imgs)