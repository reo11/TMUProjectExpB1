# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:02:21 2018

@author: hiroki
"""

import cv2
import pandas as pd
import numpy as np

INPUT_DIR = 'Query/csv/'
df = pd.read_csv(INPUT_DIR + "featurePoint.csv")

orgImgHeight = 286
orgImgWidth = 384

noseColStart = 55
noseColEnd = 73
#print(df.iloc[0, :])
df_out = df.copy()
for i in range(df.shape[0]):
    #print(i)
    imgArray = np.zeros((orgImgHeight, orgImgWidth), np.uint8)

    targetNose = df.iloc[i, noseColStart:noseColEnd]
    for j in range(0,targetNose.shape[0],2):
        imgArray[targetNose[j+1], targetNose[j]] = 255
    
    # check center of gravity
    """
    cv2.circle(imgArray, (x,y), 4, 100, 2, 4)
    cv2.imshow("test", imgArray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break
    """
    
    mu = cv2.moments(imgArray, False)
    x,y= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
    
    df_out.iloc[i, list(range(1, df.shape[1]-1, 2))] = df.iloc[i, list(range(1, df.shape[1]-1, 2))] - x
    df_out.iloc[i, list(range(2, df.shape[1]-1, 2))] = df.iloc[i, list(range(2, df.shape[1]-1, 2))] - y
    #print(x, y)    
    #print(df_out.iloc[i, :])    
    
df_out.to_csv(INPUT_DIR + "featurePoint_nosevec.csv", index=False)
    