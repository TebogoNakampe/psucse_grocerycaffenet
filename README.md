PennState 2016FA CSE586 Project2: Fine-tuning CNN <br/> - Jinhang Choi, Jihyun Ryoo
============

This webpage is a comprehensive report of the second project in CSE586 class.

Table of contents :

  * Introduction
  * Background
  * Fine-tuning CaffeNet
  * Result
  * Summary

Introduction
------------

In this project, we employ Caffe [[1](#Jia14)] as a framework for convolutional neural network (CNN).
Also, as a case study of fine-tuing CNN, we bootstrap CaffeNet [[3](#BAIRCaffeNet)], which is a replication of AlexNet [[2](#Alex12)]. We call the fine-tuned model GroceryCaffeNet because it exploits CaffeNet's capability to identify some grocery items. Training GroceryCaffeNet is based on [ImageNet](http://image-net.org/) Dataset.

Background
------------

### Caffe

Caffe is a framework for machine learning which is developed by Berkeley AI Research (BAIR).

It has numerous implementation of general layers used in CNNs; like convolution, response normalization, maximum/average pooling, rectified linear activation (relu), sigmoid, etc.
Using the layers, user can generate their own model.
Or Caffe provides several CNN references - AlexNet, GoogleNet, CaffeNet, etc. - so user can choose one of them.
In this project, we choosed CaffeNet.

Caffe support CPU (C++) and GPU (CUDA), thus user can choose what they want.


### AlexNet and CaffeNet



### ImageNet



Fine-tuning CaffeNet
------------
For target data, we decided to use ILSVRC2014 DET Dataset [[4](#ilsvrc14)] since it does not overlap ILSVRC2012 Dataset used in CaffeNet. To be specific, we choose 21 of 200 DET classes, which might represent some items in a grocery store. The following table describes a breakdown of image counts in our dataset. Due to annotation issue, all 15,380 images come from DET training dataset, where training/validation dataset are randomly separated in 9:1 manner. For the sake of simplicity, we adapted centered 255x255 downscaling to each image.

|         | apple   | artichoke | bagel   | banana  | bell pepper | burrito | cucumber  | fig     | guacamole |
|:-------:|:-------:|:---------:|:-------:|:-------:|:-----------:|:-------:|:---------:|:-------:|:--------:|
| train   | 1081    | 809       | 586     | 733     | 633         | 560     | 492       | 475     | 651       |
| val     | 140     | 95        | 87      | 88      | 68          | 84      | 77        | 53      | 90        |
| total   | 1221    | 904       | 673     | 821     | 701         | 644     | 569       | 528     | 741       |

|         | hamburger | lemon   | milk can  | mushroom | pineapple | pizza   | pomegranate | popsicle | pretzel |
|:-------:|:---------:|:-------:|:-------:|:--------:|:---------:|:-------:|:-----------:|:--------:|:------:|
| train   | 555       | 662     | 501       | 671      | 530       | 730     | 822         | 592      | 571     |
| val     | 72        | 86      | 73        | 75       | 64        | 104     | 79          | 87       | 66      |
| total   | 627       | 748     | 574       | 746      | 594       | 834     | 901         | 679      | 637     |

|         | strawberry | water | wine  | summary |
|:-------:|:----------:|:-----:|:-----:|:-------:|
| train   | 576        | 686   | 733   | 13649   |
| val     | 83         | 87    | 73    | 1731    |
| total   | 659        | 773   | 806   | 15380   |

As shown in the following CNN scheme, we directly borrowed layers of CaffeNet except softmax, which is composed of 21 classes. Even though reusing weights and biases in CaffeNet, however, we still need to refine the trained parameters for extracting features in some way. As a result, we did not eliminate learning rate of every layers, but fortify training the last fully connected layer [ten times](model/mdl_grocery_caffenet/train_val.prototxt#L364-L371) than other layers. Instead, the initial learning rate decreases to [0.001](model/mdl_grocery_caffenet/solver.prototxt#L4) so that GroceryCaffeNet can rely on the original CaffeNet's parameters. 

![train-grocery-caffenet](result/train.png)

Since there are 13,649 images in our training dataset, an epoch is roughly 110 iterations in terms of batch size 125. We repeated testing validation dataset for every 110 iterations to understand a trend of top-1 classification accuracy while training continues. Testing itself takes 28 iterations with batch size 62.

<div class="fig figcenter fighighlight">
  <img src="result/train_loss.png" width="48%">
  <img src="result/test_accuracy.png" width="48%" style="border-left: 1px solid black;">
</div>

A trend of classification accuracy in testing validation dataset indicates that GroceryCaffeNet introduces fast convergence even with just 1 epoch due to derivation of parameters from CaffeNet. In the meantime, we can ensure potential of over-fitting issue by monitoring a trend of loss penalty. From our experiments, 45,000 iterations are sufficient to probe the extent of available classification in the current setting. Our model may classify 21 grocery items in 76.2% accuracy. 

Result
------------
Let's analyze 76.2% accuracy in more detail. To understand what happend to testing validation dataset, we deploy GroceryCaffeNet on classification [script](script/classify.py) per image. For your information, please refer to the following [log](result/classification.log).

![deploy-grocery-caffenet](result/deploy.png)

Summary
------------
[PSU CSE586 Course Project2] Fine-tuning AlexNet from Selected ILSVRC14 Dataset

References
------------
<a name='Jia14'> </a>
[1] [Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, Grevor Darrell, "Caffe: Convolutional Architecture for Fast Feature Embedding", arXiv preprint arXiv:1408.5093, 2014.](http://caffe.berkeleyvision.org/ "Caffe Homepage")

<a name='Alex12'> </a>
[2] [Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", NIPS, 2012.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks "AlexNet")

<a name='BAIRCaffeNet'> </a>
[3] [Berkeley AI Research (BAIR) CaffeNet Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)

<a name='ilsvrc14'> </a>
[4] [Imagenet Large Scale Visual Recognition Challenge 2014 (ILSVRC2014) Dataset](http://image-net.org/challenges/LSVRC/2014/index#data)
