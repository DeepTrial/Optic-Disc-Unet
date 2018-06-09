import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import morphology, measure
import math
import sys
sys.path.insert(0, './cognition/lib/')


#调整图片大小
def imgResize(image,scale):
    dim = (int(image.shape[1] *scale), int(image.shape[0] * scale))
    print(dim)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR )
    return resized

#形态学处理
class morph:
    def __init__(self,ksize=5): #形态学算子
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    def setKernel(self,ksize):
        self.kernel= cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

    def close(self,image,iter=1): #闭运算
        closed=image
        for i in range(iter):
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, self.kernel)
        return closed

    def open(self,image): #开运算
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
        return opened

    def erode(self,image): #腐蚀
        eroded = cv2.erode(image, self.kernel)
        return eroded

    def dilate(self,image): #膨胀
        dilated=cv2.dilate(image,self.kernel)
        return dilated

    def medianBlur(self,image,ksize): #中值滤波
        result = cv2.medianBlur(image,ksize)
        return result

    def homofilter(self,image): #同态滤波
        img = (image - np.mean(image)) / 255
        I_log = (np.log(img + 1))
        blur = self.medianBlur(I_log.astype('uint8'), 5)
        result_log = np.log(255) - blur
        result = (np.exp(0.5 * result_log + 0.5 * I_log))
        result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
        return result

def countWhite(image): #统计二值图中白色区域面积
    return np.count_nonzero(image)

def connectTable(image,min_size,connect):
    label_image = measure.label(image)
    dst = morphology.remove_small_objects(label_image, min_size=min_size, connectivity=connect)
    return dst,measure.regionprops(dst)

def spatial_otsu(Myimg,gridx,gridy):
    patchx_add=0
    patchy_add=0
    if Myimg.shape[0]%gridx!=0:
        patchx_add=gridx-Myimg.shape[0]%gridx
    if Myimg.shape[1]%gridy!=0:
        patchy_add=gridy-Myimg.shape[1]%gridy
    new_img=np.zeros((Myimg.shape[0]+patchx_add,Myimg.shape[1]+patchy_add))
    new_img[:Myimg.shape[0],:Myimg.shape[1]]=Myimg
    new_img=(new_img).astype(np.uint8)
    gridxnum=new_img.shape[0]/gridx
    gridynum=new_img.shape[1]/gridy

    for i in range(int(gridxnum)):
        for j in range(int(gridynum)):
            patch=new_img[i*gridx:(i+1)*gridx,j*gridy:(j+1)*gridy]
            _, dst = cv2.threshold(patch, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
            new_img[i * gridx:(i + 1) * gridx, j * gridy:(j + 1) * gridy]=dst
    return new_img[:Myimg.shape[0],:Myimg.shape[1]]

def toBinary(image,thresh):
    _, Result = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return Result




def postprocess(probResult,probImage):
    dst,regionprops=connectTable(probResult,4000,1)
    result=np.zeros_like(probResult)
    prob=np.zeros_like(probResult)
    candidates = []   #被选择区域集
    dsolve=morph((5,5))
    probResult=dsolve.close(probResult,iter=2)
    for region in regionprops:  # 循环得到每一个连通区域属性集
        minr, minc, maxr, maxc = region.bbox
        area = (maxr - minr) * (maxc - minc)   #候选区域面积
        if math.fabs((maxr - minr) / (maxc - minc)) > 2.5 or math.fabs((maxr - minr) / (maxc - minc)) < 0.45 or area * 0.35 > countWhite(probResult[minr:maxr, minc:maxc]):
            #剔除细、长区域和太过夸张的内凹型、外凸形
            continue
    #筛选过的区域与已选择区域合
        candidates.append(region.bbox)
    #从原图中切割选择的区域
    savemark = 0
    for candi in range(len(candidates)):
        minr, minc, maxr, maxc = candidates[candi]
        result[minr :maxr , minc :maxc] = probResult[minr :maxr , minc :maxc]
        prob[minr :maxr , minc :maxc] = probImage[minr :maxr , minc :maxc]
    return result,prob

