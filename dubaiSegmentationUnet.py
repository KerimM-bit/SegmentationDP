# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:40:18 2023

@author: User
"""

# The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites
#  and annotated with pixel-wise semantic segmentation in 6 classes. 
#  The total volume of the dataset is 72 images grouped into 6 larger tiles. 
#  The classes are:

# Building: #3C1098
# Land (unpaved area): #8429F6
# Road: #6EC1E4
# Vegetation: #FEDD3A
# Water: #E2A929
# Unlabeled: #9B9B9B

# Images come in many sizes: 797x644, 
# 509x544, 682x658, 1099x846, 1126x1058, 
# 859x838, 1817x2061,  2149x1479​

import os
import cv2
import numpy as np

import tensorflow as tf
print("Num GPUS available ", len(tf.config.experimental.list_physical_devices('GPU')))

"""

2023-02-21 19:24:31.564107: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow 
    binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU 
    instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

"""
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import tensorflow.keras as keras
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU

from sklearn.preprocessing import MinMaxScaler, StandardScaler


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
# Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
# Memory growth must be set before GPUs have been initialized
        print(e)


scaler = MinMaxScaler()

root_directory = 'Semantic segmentation dataset/'

patch_size = 128

# Read images from respective subdirectory
# Since images are different size, I will crop them nearest size divisible by 128
# and then divide all images into pacthes of 128 x 128

image_dataset = []
for path, subdirs, files in os.walk(root_directory):
    #print(files)
    dirname = path.split(os.path.sep)[-1]
    #print(dirname)
    if dirname == 'images': #Find all 'images' directories
        images = os.listdir(path)  #list all image name in this subdirectory
        for i, image_name in enumerate(images):
            if image_name.endswith(".jpg"): #Only read jpg images.
                
                image = cv2.imread(path+"/"+image_name, 1) #Read image as BGR
                SIZE_X = (image.shape[1]//patch_size) * patch_size #Nearest size divisible by patch size
                SIZE_Y = (image.shape[0]//patch_size) * patch_size #Nearest size divisible by patch size
                image = Image.fromarray(image) 
                image = image.crop((0, 0, SIZE_X, SIZE_Y)) #crop from the top left corner
                image = np.array(image) 
                
                # Extract patches from each image
                print("Now patchifying image: ", path+"/"+image_name)
                patched_img = patchify(image
                                       , (patch_size, patch_size, 3),
                                       step=patch_size)
                for i in range(patched_img.shape[0]):
                    for j in range(patched_img.shape[1]):
                    
                        single_patch_img = patched_img[i, j, :, :]
                        # Using minmaxscaler instead of just dividing by 255
                        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        
                        #single_patch_img = (single_patch_img.astype('float32'))
                        single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.   
                        image_dataset.append(single_patch_img)
                        
                        
#Same process as above for masks
mask_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':
        masks = os.listdir(path)
        for i, mask_name in enumerate(masks):
            if mask_name.endswith('.png'):
                
                mask = cv2.imread(path+'/'+mask_name, 1)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1]//patch_size) * patch_size
                SIZE_Y = (mask.shape[0]//patch_size) * patch_size
                mask = Image.fromarray(mask)
                mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
                mask = np.array(mask)
                
                print("Now patcifying mask: ", path+"/"+mask_name)
                patched_mask = patchify(mask,
                                        (patch_size, patch_size, 3),
                                        step=patch_size)
                for i in range(patched_mask.shape[0]):
                    for j in range(patched_mask.shape[1]):
                        
                        single_patch_mask = patched_mask[i, j, :, :]
                        #single_patch_img = (single_patch_img.astype('float32'))
                        single_patch_mask = single_patch_mask[0]
                        mask_dataset.append(single_patch_mask)
                        
                        
image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)


    

"""
RGB to HEX (Hexademical: bsse 16)
This number divided by sixteen (integer division; ignoring any remainder) gives 
the first hexadecimal digit (between 0 and F, where the letters A to F represent 
the numbers 10 to 15). The remainder gives the second hexadecimal digit. 
0-9 --> 0-9
10-15 --> A-F


Example: RGB --> R=201, G=, B=
R = 201/16 = 12 with remainder of 9. So hex code for R is C9 (remember C=12)

Calculating RGB from HEX: #3C1098
3C = 3*16 + 12 = 60
10 = 1*16 + 0 = 16
98 = 9*16 + 8 = 152
"""
#Convert HEX to RGB array
# Try the following to understand how python handles hex values...
a = int('3C', 16)
print(a)

# Doing the same for all RGB channels in each hex code to convert them to RGB
Building = '#3C1098'.lstrip('#')
Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) #60, 16, 152

Land = '#8429F6'.lstrip('#')
Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246

Road = '#6EC1E4'.lstrip('#')
Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228

Vegetation = 'FEDD3A'.lstrip("#")
Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58

Water = 'E2A929'.lstrip("#")
Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41

Unlabeled = '#9B9B9B'.lstrip('#')
Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) # 155, 155, 155

label = single_patch_mask


# Now replace RGB to integer values to be used as labels.
# Find pixels with combination of RGB for the above defined arrays...
# If matches then replace all values in that pixel with a specific integer

def rgb_to_2D_label(label):
    """
    
    Suply our labels masks as input in RGB format
    Replace pixels with specific RGB values
    
    """
    label_seg = np.zeros(label.shape, dtype=np.uint8)
    label_seg[np.all(label == Building, axis = -1)] = 0
    label_seg[np.all(label == Land, axis = -1)] = 1 
    label_seg[np.all(label == Road, axis = -1)] = 2 
    label_seg[np.all(label == Vegetation, axis = -1)] = 3 
    label_seg[np.all(label == Water, axis = -1)] = 4 
    label_seg[np.all(label == Unlabeled, axis = -1)] = 5
    
    
    label_seg = label_seg[:, :, 0] # Just take the first channel, no need for all 3 channels
    
    return label_seg

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)
    
labels = np.array(labels)
labels = np.expand_dims(labels, axis = 3)

print("Unique labels in label dataset are: ", np.unique(labels))
    
############################################################################    
    
#Another Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:,:,0])
plt.show()    
    
     
n_classes = len(np.unique(labels))
from keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state = 42)




weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights = weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]


from simple_multi_unet_model import multi_unet_model, jacard_coef

metrics = ['accuracy', jacard_coef]

def get_model():
    return multi_unet_model(n_classes=n_classes, 
                            IMG_HEIGHT=IMG_HEIGHT,
                            IMG_WIDTH=IMG_WIDTH,
                            IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
model.summary()

#In order to not train model everytime we run the script, I will save the model and use that
#for either further training or predictions


#Model Checkpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint("models/satellite_standard_unet_100epochs.hdf5", 
                                                  verbose=1, save_best_only=True)

call_backs = [
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_lose'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]



history1 = model.fit(X_train, y_train,
                      batch_size = 16,
                      callbacks = call_backs,
                      epochs = 100,
                      validation_data=(X_test, y_test),
                      shuffle=False)

###########################################################
#plot the training and validation accuracy and loss at each epoch
history = history1
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['jacard_coef']
val_acc = history.history['val_jacard_coef']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()


from keras.models import load_model

model = load_model("models/satellite_standard_unet_100epochs.hdf5",
                   custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                   'jacard_coef': jacard_coef})


# from tensorflow.keras.models import load_model
# model = load_model("models/satellite_standard_unet_100epochs.hdf5",
#                    custom_objects={"dice_loss_plus_2focal_loss": total_loss,
#                                    'jacard_coef':jacard_coef})
# IOU

y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_pred, axis=3)


# Using built in keras function for IoU

from tensorflow.keras.metrics import MeanIoU
n_classes = 6 
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU = ", IOU_keras.result().numpy())

############################################################
#Predict on a few images
import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test_argmax[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()






 






















    