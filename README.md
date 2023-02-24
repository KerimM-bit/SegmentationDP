# SegmentationDP

## In this project with the data I downloaded from kaggle I tried perform  image segmentation. I used U-net for segmentation. 

Briefly about the data I will be working on: !(https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)

The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes. The total volume of the dataset is 72 images grouped into 6 larger tiles. The classes are:

#### 1 Building: #3C1098
#### 2 Land (unpaved area): #8429F6
#### 3 Road: #6EC1E4
#### 4 Vegetation: #FEDD3A
#### 5 Water: #E2A929
#### 6 Unlabeled: #9B9B9B

As you can see, classes are given in HES values, so it is necessary to do some kind of preprocessing and get those HES values to their respective RGB value

Before starting to train the the model, it is best practice to check whether all that preprocessing corrupted images in any way, if it has, then before feeding them into the model they need to be fixed again so model works/performs properly.
After everything checks out,  I can precede.
## Accuracy of the model
Due to limited amount of data in this project accuracy of the model is around 0.70, which is to be honest not really bad result, but of course with different accuracy metrics and approaches it is possible for  model to perform even better. 

![Accuracy of the model](https://github.com/KerimM-bit/SegmentationDP/blob/master/results_img/Figure%202023-02-24%20175728.png)

 
