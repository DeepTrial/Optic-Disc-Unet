#################
# Resnick Xing
# github @DeepTrial
# 2018/4/7
#################
#from .basic import *
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from lib.imagic import *
from lib.help_functions import *

class patchClsGen:

	def __init__(self,backend,imgPack,maskPack,batchSize,epochs,class_weight,**kwargs):
		self.backend=backend
		self.imgs=imgPack
		self.mask=maskPack
		self.batch_size=batchSize
		self.epochs=epochs
		self.class_weight=class_weight
		if kwargs.get('patch_height',None)!=None:
			self.patch_h=kwargs['patch_height']
		if kwargs.get('patch_width',None)!=None:
			self.patch_w=kwargs['patch_width']
		if kwargs.get('subimgs',None)!=None:
			self.subimgs=np.array(kwargs['subimgs'])
		if kwargs.get('Nimgs',None)!=None:
			self.Nimgs=np.array(kwargs['Nimgs'])
		if kwargs.get('num_lesion_class',None)!=None:
			self.num_lesion_class=np.array(kwargs['num_lesion_class'])

	def _sampler(self):
		img_h = self.imgs.shape[2]
		img_w = self.imgs.shape[3]

		# p = random.uniform(0, 1)
		# psum = 0
		# label = 0
		# self.class_weight = self.class_weight / np.sum(self.class_weight)
		# for i in range(self.class_weight.shape[0]):
		# 	psum = psum + self.class_weight[i]
		# 	if p < psum:
		# 		label = i
		# 		break
		label=1
		mlist = [np.where(self.mask[:, 0, :, :] == np.max(self.mask[:, 0, :, :]))]
		imgIndex=random.randint(0,self.Nimgs-1)
		x_center = random.randint(1 + int(self.patch_w / 2), img_w - int(self.patch_w / 2))
		y_center = random.randint(1 + int(self.patch_h / 2), img_h - int(self.patch_h / 2))
		#imgPatch=self.imgs[imgIndex,0,int(y_center - self.patch_h / 2):int(y_center + self.patch_h / 2),int(x_center - self.patch_w / 2):int(x_center + self.patch_w / 2)]
		maskPatch=self.mask[imgIndex,0,int(y_center - self.patch_h / 2):int(y_center + self.patch_h / 2),int(x_center - self.patch_w / 2):int(x_center + self.patch_w / 2)]

		while maskPatch[int(self.patch_h/2),int(self.patch_w/2)]!=1 and countWhite(maskPatch)<=self.patch_w*self.patch_h*0.5:
			t = mlist[label-1]
			cid = random.randint(0, t[0].shape[0] - 1)
			imgIndex = t[0][cid]
			y_center = t[1][cid] + random.randint(0 - int(self.patch_w / 2), 0 + int(self.patch_w / 2))
			x_center = t[2][cid] + random.randint(0 - int(self.patch_w / 2), 0 + int(self.patch_w / 2))
			if y_center < self.patch_w / 2:
				y_center = self.patch_w / 2
			elif y_center > img_h - self.patch_w / 2:
				y_center = img_h - self.patch_w / 2

			if x_center < self.patch_w / 2:
				x_center = self.patch_w / 2
			elif x_center > img_w - self.patch_w / 2:
				x_center = img_w - self.patch_w / 2
			#imgPatch = self.imgs[imgIndex, 0, int(y_center - self.patch_h / 2):int(y_center + self.patch_h / 2),int(x_center - self.patch_w / 2):int(x_center + self.patch_w / 2)]
			maskPatch = self.mask[imgIndex, 0, int(y_center - self.patch_h / 2):int(y_center + self.patch_h / 2),int(x_center - self.patch_w / 2):int(x_center + self.patch_w / 2)]

		return imgIndex,x_center, y_center,label

	def active(self):
		while 1:
			img_h = self.imgs.shape[2]
			img_w = self.imgs.shape[3]
			for t in range(int(self.subimgs * self.Nimgs / self.batch_size)):
				X = np.zeros([self.batch_size, 1, self.patch_h, self.patch_w])
				Y = np.zeros([self.batch_size,self.patch_h*self.patch_w,self.num_lesion_class+1])
				for j in range(self.batch_size):
					[i_center, x_center, y_center,label] = self._sampler()
					patch = self.imgs[i_center, :, int(y_center - self.patch_h / 2):int(y_center + self.patch_h/2 ),int(x_center - self.patch_w / 2):int(x_center + self.patch_w/2)]
					patch_mask =np.reshape(self.mask[i_center, :, int(y_center - self.patch_h / 2):int(y_center + self.patch_h / 2),int(x_center - self.patch_w / 2):int(x_center + self.patch_w / 2)],[1, self.patch_h, self.patch_w,self.class_weight.shape[0]])
					if patch.shape[1]!=self.patch_h or patch.shape[2]!=self.patch_w:
						print('\n',label,int(y_center - self.patch_h / 2),int(y_center + self.patch_h/2 ),int(x_center - self.patch_w / 2),int(x_center + self.patch_w / 2),y_center,x_center)
					X[j, :, :, :] = patch
					Y[j, :,:] =masks_Unet(np.reshape(patch_mask, [1, self.num_lesion_class, self.patch_h, self.patch_w]), self.num_lesion_class)
				yield (X, Y)
