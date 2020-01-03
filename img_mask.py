import numpy as np  # linear algebra
import pandas as pd
import matplotlib.pylab as plt
import os
import cv2
from config import Config
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv(Config.DATA_PATH + '/train.csv')

train_imagesfolder = os.listdir(Config.DATA_PATH + "/train_images")  # dir is your directory path
trainimagesfilecount = len(train_imagesfolder)

train_masksfolder = os.listdir(Config.DATA_PATH + r"train_masks")  # dir is your directory path
trainmasksfilecount = len(train_masksfolder)


def CreateMaskImages(imageName):
    trainimage = cv2.imread(Config.DATA_PATH + "/train_images/" + imageName + '.jpg')
    imagemask = cv2.imread(Config.DATA_PATH + "/train_masks/" + imageName + ".jpg", 0)
    try:
        imagemaskinv = cv2.bitwise_not(imagemask)
        res = cv2.bitwise_and(trainimage, trainimage, mask=imagemaskinv)
        plt.imshow(imagemask)
        cv2.imwrite(Config.DATA_PATH + "MaskTrain/" + imageName + ".jpg", res)
    except:
        print("exception for image" + imageName)
        cv2.imwrite(Config.DATA_PATH + "MaskTrain/" + imageName + ".jpg", trainimage)


for i in range(len(train_data)):
    ImageName = train_data.loc[i, "ImageId"]
    print(ImageName)
    CreateMaskImages(ImageName)
