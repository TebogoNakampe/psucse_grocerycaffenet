PSU CSE586 Project2 [GroceryCaffeNet] - Jinhang Choi, Jihyun Ryoo
============

This webpage is a report of CSE586 class, second project.

Table of contents :

  * Introduction
  * Background
  * Fine-tuning of CaffeNet
  * Result
  * Summary

Introduction
------------

In this project, we used Caffe [1] which is a deep learning framework for convolutional neural network.
We used AlexNet 

Background
------------




Fine-tuning of CaffeNet
------------

|         | apple   | artichoke | bagel   | banana  | bell    | burrito | cucumber  |
|:-------:|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:---------:|
| train   | 1081    | 809       | 586     | 733     | 633     | 560     | 492       |
| val     | 140     | 95        | 87      | 88      | 68      | 84      | 77        |
| total   | 1221    | 904       | 673     | 821     | 701     | 644     | 569       |

|         | fig     | guacamole | hamburger | lemon   | milk    | mushroom | pineapple |
|:-------:|:-------:|:---------:|:---------:|:-------:|:-------:|:--------:|:---------:|
| train   | 475     | 651       | 555       | 662     | 501     | 671      | 530       |
| val     | 53      | 90        | 72        | 86      | 73      | 75       | 64        |
| total   | 528     | 741       | 627       | 748     | 574     | 746      | 594       |

|         | pizza   | pomegranate | popsicle | pretzel | strawberry | water | wine  | summary |
|:-------:|:-------:|:-----------:|:--------:|:-------:|:----------:|:-----:|:-----:|:-------:|
| train   | 730     | 822         | 592      | 571     | 576        | 686   | 733   | 13649   |
| val     | 104     | 79          | 87       | 66      | 83         | 87    | 73    | 1731    |
| total   | 834     | 901         | 679      | 637     | 659        | 773   | 806   | 15380   |

<div class="fig figcenter fighighlight">
  <img src="result/train_loss.png" width="48%">
  <img src="result/test_accuracy.png" width="48%" style="border-left: 1px solid black;">
</div>

Result
------------

Summary
------------
[PSU CSE586 Course Project2] Fine-tuning AlexNet from Selected ILSVRC14 Dataset

References
------------
[1] [http://caffe.berkeleyvision.org/](http://caffe.berkeleyvision.org/ "Caffe Homepage")

[2] [Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", NIPS, 2012.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks "AlexNet")