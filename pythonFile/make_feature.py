# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:02:21 2018

@author: hiroki
"""

#import cv2
import pandas as pd
import numpy as np
#import gc

faceStart = 1
faceEnd = 138

chinColStart = 1
chinColEnd = 35

rightEyebrowStart = 35
rightEyebrowEnd = 45

leftEyebrowStart = 45
leftEyebrowEnd = 55

noseColStart = 55
noseColEnd = 73

rightEyeStart = 73
rightEyeEnd = 85

leftEyeStart = 85
leftEyeEnd = 97

mouseStart = 97
mouseEnd = 138

for mode in ['DB', 'Query']:
    INPUT_DIR = '../DlibDataChangeLuminace/' + mode + '/csv/'
    df = pd.read_csv(INPUT_DIR + "featurePoint.csv")

    df_out = pd.DataFrame(df['target'].copy())

    # face(顔)
    df_target = df.iloc[:, faceStart:faceEnd]
    df_target_x = df_target.iloc[:, 0:-1:2]
    df_out['face_width'] = df_target_x.max(axis=1) - df_target_x.min(axis=1)
    df_target_y = df_target.iloc[:, 1:-1:2]
    df_out['face_height'] = df_target_y.max(axis=1) - df_target_y.min(axis=1)

    # chin(アゴ)
    df_target = df.iloc[:, chinColStart:chinColEnd]
    df_target_x = df_target.iloc[:, 0:-1:2]
    df_out['chin_width'] = df_target_x.max(axis=1) - df_target_x.min(axis=1)
    df_target_y = df_target.iloc[:, 1:-1:2]
    df_out['chin_height'] = df_target_y.max(axis=1) - df_target_y.min(axis=1)

    # right eye brow(右マユ)
    df_target = df.iloc[:, rightEyebrowStart:rightEyebrowEnd]
    df_target_x = df_target.iloc[:, 0:-1:2]
    df_out['right_Eyebrow_width'] = df_target_x.max(axis=1) - df_target_x.min(axis=1)
    df_target_y = df_target.iloc[:, 1:-1:2]
    df_out['right_Eyebrow_height'] = df_target_y.max(axis=1) - df_target_y.min(axis=1)

    # left eye brow(左マユ)
    df_target = df.iloc[:, leftEyebrowStart:leftEyebrowEnd]
    df_target_x = df_target.iloc[:, 0:-1:2]
    df_out['left_Eyebrow_width'] = df_target_x.max(axis=1) - df_target_x.min(axis=1)
    df_target_y = df_target.iloc[:, 1:-1:2]
    df_out['left_Eyebrow_height'] = df_target_y.max(axis=1) - df_target_y.min(axis=1)

    # nose(鼻)
    df_target = df.iloc[:, noseColStart:noseColEnd]
    df_target_x = df_target.iloc[:, 0:-1:2]
    df_out['nose_width'] = df_target_x.max(axis=1) - df_target_x.min(axis=1)
    df_target_y = df_target.iloc[:, 1:-1:2]
    df_out['nose_height'] = df_target_y.max(axis=1) - df_target_y.min(axis=1)
    df['nose_width_center'] = df_target_x.mean(axis=1)
    df['nose_height_center'] = df_target_y.mean(axis=1)

    # right eye(右目)
    df_target = df.iloc[:, rightEyeStart:rightEyeEnd]
    df_target_x = df_target.iloc[:, 0:-1:2]
    df_out['right_Eye_width'] = df_target_x.max(axis=1) - df_target_x.min(axis=1)
    df_target_y = df_target.iloc[:, 1:-1:2]
    df_out['right_Eye_height'] = df_target_y.max(axis=1) - df_target_y.min(axis=1)
    df['right_Eye_width_center'] = df_target_x.mean(axis=1)
    df['right_Eye_height_center'] = df_target_y.mean(axis=1)

    # left eye(左目)
    df_target = df.iloc[:, leftEyeStart:leftEyeEnd]
    df_target_x = df_target.iloc[:, 0:-1:2]
    df_out['left_Eye_width'] = df_target_x.max(axis=1) - df_target_x.min(axis=1)
    df_target_y = df_target.iloc[:, 1:-1:2]
    df_out['left_Eye_height'] = df_target_y.max(axis=1) - df_target_y.min(axis=1)
    df['left_Eye_width_center'] = df_target_x.mean(axis=1)
    df['left_Eye_height_center'] = df_target_y.mean(axis=1)

    # mouse(口)
    df_target = df.iloc[:, mouseStart:mouseEnd]
    df_target_x = df_target.iloc[:, 0:-1:2]
    df_out['mouse_width'] = df_target_x.max(axis=1) - df_target_x.min(axis=1)
    df_target_y = df_target.iloc[:, 1:-1:2]
    df_out['mouse_height'] = df_target_y.max(axis=1) - df_target_y.min(axis=1)
    df['mouse_width_center'] = df_target_x.mean(axis=1)
    df['mouse_height_center'] = df_target_y.mean(axis=1)

    # relative distance(各部位の相対的な距離)
    df_out['eye2eye_dist'] = np.sqrt((df['right_Eye_width_center'] - df['left_Eye_width_center'])**2 + \
                                (df['right_Eye_height_center'] - df['left_Eye_height_center'])**2)
    df_out['Reye2nose_dist'] = np.sqrt((df['right_Eye_width_center'] - df['nose_width_center'])**2 + \
                                (df['right_Eye_height_center'] - df['nose_height_center'])**2)
    df_out['Leye2nose_dist'] = np.sqrt((df['left_Eye_width_center'] - df['nose_width_center'])**2 + \
                                (df['left_Eye_height_center'] - df['nose_height_center'])**2)
    df_out['nose2mouse_dist'] = np.sqrt((df['nose_width_center'] - df['mouse_width_center'])**2 + \
                                (df['nose_height_center'] - df['mouse_height_center'])**2)
    df_out['nose2mouse_dist'] = np.sqrt((df['nose_width_center'] - df['mouse_width_center'])**2 + \
                                (df['nose_height_center'] - df['mouse_height_center'])**2)
    df_out['Reye2mouse_dist'] = np.sqrt((df['right_Eye_width_center'] - df['mouse_width_center'])**2 + \
                                (df['right_Eye_height_center'] - df['mouse_height_center'])**2)
    df_out['Leye2mouse_dist'] = np.sqrt((df['left_Eye_width_center'] - df['mouse_width_center'])**2 + \
                                (df['left_Eye_height_center'] - df['mouse_height_center'])**2)

    for i in range(df_out.shape[0]):
        df_out.iloc[i, 1:] = df_out.iloc[i, 1:]/df_out['face_width'][i]
    df_out.drop('face_width', axis=1, inplace=True)

    df_out.to_csv(INPUT_DIR + "features_rel_dist.csv")
