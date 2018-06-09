import os,numpy as np,cv2,h5py,configparser,glob
from PIL import Image
from math import ceil
import matplotlib.pyplot as plt
from lib.help_functions import *

#-----------------------load settings-------------------------------------------------------
config = configparser.RawConfigParser()
config.read('./configuration.txt')

N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = config.get('data paths', 'original_imgs_train')
groundTruth_imgs_train = config.get('data paths', 'groundTruth_imgs_train')
#num_lesion_class = int(config.get('data attributes', 'num_lesion_class'))

#test
original_imgs_test = config.get('data paths', 'original_imgs_test')
groundTruth_imgs_test = config.get('data paths', 'groundTruth_imgs_test')

#save
dataset_path = config.get('data paths','dataset_path')
#------------Confirm image type and nums-------------------------------------------------------
print("HINT: all images must have the same size!")

imgList = glob.glob(original_imgs_train + '/*.jpg')
Nimgs_train=len(imgList)
config.set('data attributes','total_data_train',str(Nimgs_train))

imgList = glob.glob(original_imgs_test + '/*.jpg')
Nimgs_test=len(imgList)
config.set('data attributes','total_data_test',str(Nimgs_test))

#------------Preload to know image size--------------------------------------------------------
img2test=cv2.imread(imgList[0])
channels = 1# img2test.shape[2]
height = img2test.shape[0]
width = img2test.shape[1]
config.set('data attributes','height',str(height))
config.set('data attributes','width',str(width))
config.write(open('configuration.txt','w'))


def get_datasets(imgs_dir,groundTruth_dir,Nimgs,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,1,height,width))

    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print("original image: " +files[i])
            img = plt.imread(imgs_dir+files[i])
            img = cv2.resize(img,(width,height))
            imgG=img[:,:,1]*0.75+img[:,:,0]*0.25
            imgG=np.reshape(imgG,(height,width,1))
            img=imgG
            imgs[i] = np.asarray(img)
            #corresponding groundtruth

            groundTruth_name = files[i]
            print("groundtruth name: %s"%(groundTruth_name))
            g_truth = Image.open(groundTruth_dir + groundTruth_name).resize([width,height],Image.BICUBIC).convert('L')
            groundTruth[i,0,:,:] = np.asarray(g_truth)
            #corresponding border masks
    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255 )
    assert(np.min(groundTruth)==0 )
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    return imgs, groundTruth




imgs_train, groundTruth = get_datasets(original_imgs_train,groundTruth_imgs_train,Nimgs_train,"train")
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + config.get("data paths","train_imgs_original"))#"DRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth, dataset_path + config.get("data paths","train_groundtruth"))#"DRIVE_dataset_groundTruth_train.hdf5")


imgs_test, groundTruth = get_datasets(original_imgs_test,groundTruth_imgs_test,Nimgs_test,"test")
print("saving test datasets")
write_hdf5(imgs_test, dataset_path + config.get("data paths","test_imgs_original"))#"DRIVE_dataset_imgs_test.hdf5")
write_hdf5(groundTruth, dataset_path + config.get("data paths","test_groundtruth"))#"DRIVE_dataset_groundTruth_test.hdf5")
#
# imgs_neg, groundTruth_neg = get_datasets('./Dataset/image/train/neg_image/','./Dataset/image/train/neg_groundtruth/',11,"neg")
# print("saving neg datasets")
# write_hdf5(imgs_neg, dataset_path + "dataset_imgs_negative.hdf5")#"DRIVE_dataset_imgs_test.hdf5")
# write_hdf5(groundTruth_neg, dataset_path + "dataset_groundtruth_negative.hdf5")#"DRIVE_dataset_groundTruth_test.hdf5")

