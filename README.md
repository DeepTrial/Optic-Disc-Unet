# Optic-Disc-Unet


A modified Attention Unet model with post-process for retina optic disc segmention.

The performance of our model on Messidor-dataset：

![](https://i.imgur.com/nLMUSLd.jpg)


## Patched Based Attention Unet Model
I use a modified Attention Unet which input of model is 128x128pix image patches. To know more about attn-unet, please see the [paper][1]. When sampling the patches, I focus the algorithm get samples around optic disc. The patches is like that:

![sample patches](https://i.imgur.com/nBewGl9.jpg)

so the groundtruth is:

![](https://i.imgur.com/CKIX3tO.jpg)

## Pretrained Model & Dataset
The model is trained on DRION dataset. 90 images to train. 19 images to test.

To get the groundtruth of DRION, I write a convert tool, you can find in [DRION-DB-PythonTools][2].

Pretrained can be downloaded [here][3]. Extract them to dir `Dataset`.


## Post-Process Methods
When directly use unet model, we often get some error predictions. So I use a post-process algorithm:

1. predicted area can't be to small.
2. minimum bounding rectangle's height/width or width/height should be in 0.45~2.5

lefted area is the final output. The problem of this algorithm is that the parameters not self-adjusting, so you have to change them if input image is larger or smaller than before.

## Project Structure
The structure is based on my own [DL_Segmention_Template][4]. Difference between this project and the template is that we have metric module in dir: `perception/metric/`. To get more Information about the structure please see `readme` in [DL_Segmention_Template][4].

You can find model parameter in **configs/segmention_config.json**.

### First to run
**please run main_trainer.py first time**, then you will get data_route in `experiment` dir. Put your data in there, now you can run `main_trainer.py` again to train a model. 

### where to put Pretrained Model
The model is trained with *DRION dataset* on my own desktop (intel i7-7700hq, 24g, gtx1050 2g) within 30 minutes.
Dataset 

### Test your own image
If u want to test your own image, put your image to **(OpticDisc)/test/origin**,and change the img_type of predict settings in **configs/segmention_config.json**, run `main_test.py` to get your result. The result is in **(OpticDisc)/test/result**


[1]: https://arxiv.org/pdf/1804.03999v3.pdf
[2]: https://github.com/DeepTrial/DRION-DB-PythonTools
[3]: https://drive.google.com/open?id=10E77IiHREFKYAgkMivsIRCBsEJHjgJIz
[4]: https://github.com/DeepTrial/DL_Segmention_Template
