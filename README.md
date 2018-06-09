# Optic-Disc-Unet


A modified Unet model with post-process for retina optic disc segmention.

The performance of our model on Messidor-dataset：

![](https://i.imgur.com/nLMUSLd.jpg)


## Patched Based Attention Unet Model
I use modified Attention Unet which input is 128x128pix image patches.To know more about attn-unet,please see the [paper][1].When sampling the patches,I focus the algorithm get samples around optic disc.The patches is like that:

![sample patches](https://i.imgur.com/nBewGl9.jpg)

so the groundtruth is :

![](https://i.imgur.com/CKIX3tO.jpg)

## Pretrained Model & Dataset
the model is trained on DRION dataset.90 images to train. 19 images to test.

To get the groundtruth of DRION, I write a convert tool,you can find in [DRION-DB-PythonTools][2].

pretrained can be downloaded [here][3].extract them to dir Dataset.


## Post-Process Methods
when directly use unet model, we often get some error predictions.So I use a post-process algorithm:

1. predicted area can't be to small.
2. minimum bounding rectangle's height/width or width/height should be in 0.45~2.5

lefted area is the final output.The problem of this algorithm is that the parameters not self-adjusting.so you have to change them if input image is larger or smaller than before.

## Model Structure
the model is trained on my own desktop(intel i7-7700hq,24GB RAM,gtx1050,2GB RAM) for about 1~2 hours.I use **keras(theano backend)** to write this model.

1. **prepare_dataset.py** this file is to generate hdf5 file. The training and test dataset is in dir *Dataset/image* and generated hdf5 files store in dir *Dataset/hdf5*.
2. **xtrain.py** You can find sample-patches algorithm and trainer in this file.
3. **xpredict.py** predict images in dir *TestFold/origin* one by one
4. **odModel.py** define our model.You may find original unet,Resblock-Unet,Denseblock-Unet and AttnUnet.



[1]: https://arxiv.org/pdf/1804.03999v3.pdf
[2]: https://github.com/DeepTrial/DRION-DB-PythonTools
[3]: https://drive.google.com/open?id=1s1ri0glpr2R8bV_7iQgIBBKS1v3RDYQt