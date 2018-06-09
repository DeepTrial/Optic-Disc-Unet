import numpy as np
import configparser
from matplotlib import pyplot as plt
import cv2
import glob
from keras.models import model_from_json
from keras.models import Model
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from lib.help_functions import *
# extract_patches.py
from lib.extract_patches import recompone_overlap
from lib.extract_patches import get_data_predict_overlap
from lib.pre_processing import *
import lib.imagic as imagic



#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')
#===========================================s
#run the training on invariant or local
path_data = config.get('data paths', 'path_local')

#original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')


patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))

stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)

name_experiment = config.get('experiment name', 'name')
path_experiment = './DataSet/'
#N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
#====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')
num_lesion_class = int(config.get('data attributes','num_lesion_class'))
Nimgs = int(config.get('data attributes','total_data_test'))

imgList = glob.glob('./TestFold/origin/'+'*.'+config.get('predict settings','img_type'))

Nimgs_train=len(imgList)


for i in imgList:
    test_imgs = plt.imread(i)
    #test_imgs_original=cv2.imread(i)
    height, width = test_imgs.shape[:2]

    if height>1000 or width>1000:
        test_imgs_change=imagic.imgResize(test_imgs,0.5)
        height=int(height/2)
        width=int(width/2)
    else:
        test_imgs_change=test_imgs
    test_imgs_original=test_imgs_change[:,:,1]*0.75+test_imgs_change[:,:,0]*0.25
    _,mask=cv2.threshold(test_imgs_original,25,255,cv2.THRESH_BINARY)
    mask=mask/255
    test_imgs_original=np.reshape(test_imgs_original,(height,width,1))
    full_img_height = height#int(height*0.5)
    full_img_width = width#int(width*0.5)

    name=i.split('\\')[-1]
    picname=name.split('.')[0]

    print(picname,'start')
#============ Load the data and divide in patches
    patches_imgs_test = None
    new_height = None
    new_width = None

    patches_imgs_test, new_height, new_width,original_adjust = get_data_predict_overlap(
        imgPredict = test_imgs_original,  #original
        patch_height = patch_height,
        patch_width = patch_width,
        stride_height = stride_height,
        stride_width = stride_width,
        num_lesion = num_lesion_class,
        total_data = Nimgs
        )

#================ Run the prediction of the patches ==================================
    best_last = config.get('testing settings', 'best_last')
#Load the saved model
    model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
    model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
#Calculate the predictions
    predictions = model.predict(patches_imgs_test, batch_size=30, verbose=1)
    print("\npredicted images size :")
    print(predictions.shape)
    print(np.max(predictions[:,:,0]))
    print('test',np.max(predictions),np.min(predictions))
#===== Convert the prediction arrays in corresponding images
    pred_patches = pred_to_imgs(predictions,"original")
    pred_patches=pred_patches.transpose(0,3,1,2)
    print(pred_patches.shape)


#========== Elaborate and visualize the predicted images ====================
    pred_imgs = None
    orig_imgs = None
   
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width,patch_height,patch_width, stride_height, stride_width)# predictions
    pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]

    #visualize_all(group_images(pred_imgs[:, 0:1, :, :], N_visual), './TestFold/result/' + picname + "_prob.jpg")


    #original_adjust = original_adjust.transpose(0, 2, 3, 1)[0]
    #original_adjust=np.reshape( original_adjust,( original_adjust.shape[0], original_adjust.shape[1]))
    pred_gt_prob=pred_imgs.transpose(0,2,3,1)[0]
    # plt.imshow(pred_gt_prob[:,:,0])
    # plt.show()
    _,pred_gt=cv2.threshold(pred_gt_prob[:,:,0]*mask,0.5,1,cv2.THRESH_BINARY)   # to binary
    pred_gt,prob_gt=imagic.postprocess(pred_gt,pred_gt_prob[:,:,0])  # delete small area
    pred_gt=pred_gt.astype(np.uint8)
    _, contours, _ = cv2.findContours(pred_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # draw segmentation line
    test_imgs_change= cv2.drawContours(test_imgs_change, contours, -1, (0, 255, 0), 3)
    test_imgs_change = cv2.cvtColor(test_imgs_change, cv2.COLOR_RGB2BGR)

    cv2.imwrite('./TestFold/result/'+picname+'_merge.jpg' , test_imgs_change)
    cv2.imwrite('./TestFold/result/' + picname + '_prob.jpg', (prob_gt*255).astype(np.uint8))
