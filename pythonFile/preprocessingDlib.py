# -*- coding: utf-8 -*-
import os
import cv2
import csv
import dlib
import imutils
import numpy as np
import pandas as pd
from pathlib import Path
from imutils import face_utils
from tqdm import tqdm

DBPath = '../input/default/dataset/DB/jpeg/'
QueryPath = '../input/default/dataset/Query/jpeg/'
SIZE = (128, 128)

FILENAME = 'histFlattening'

DBSavePath = '../input/Dlib/cutface/' + FILENAME + '/DB/jpeg/'
QuerySavePath = '../input/Dlib/cutface/' + FILENAME + '/Query/jpeg/'
DBPlotImagePath = '../input/Dlib/cutface/' + FILENAME + '/DBPlot/jpeg/'
QueryPlotImagePath = '../input/Dlib/cutface/' + FILENAME + '/QueryPlot/jpeg/'
DBPlotCSVPath = '../input/Dlib/cutface/' + FILENAME + '/DB/csv/'
QueryPlotCSVPath = '../input/Dlib/cutface/' + FILENAME + '/Query/csv/'

detector = dlib.get_frontal_face_detector()
predictor_path = "./libs/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

for path in [DBSavePath, QuerySavePath, DBPlotImagePath, \
            QueryPlotImagePath, DBPlotCSVPath, QueryPlotCSVPath]:
    if not os.path.exists(path):
        os.makedirs(path)

def cutFace():
    # Database
    print('Start DB')
    p = Path(DBPath)
    p = sorted(p.glob("*.jpg"))
    df = pd.DataFrame()
    target_df = pd.DataFrame()
    preFace = None
    for index, filename in enumerate(tqdm(p)):
        img = cv2.imread(DBPath + filename.name)
        face, plotImg, featurePoint_df = face_shape_detector_dlib(img, filename.name)
        df = pd.concat([df, featurePoint_df])
        if len(face)==0:
            face = preFace
            print('db error')
        preFace = face
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.equalizeHist(face)
        face = cv2.resize(face,SIZE)
        cv2.imwrite(DBSavePath + filename.name, face)
        cv2.imwrite(DBPlotImagePath + filename.name, plotImg)

        target = int(index / 10) + 1
        t_df = pd.DataFrame({'target' : target},
                    index = [filename.name])
        target_df = pd.concat([target_df, t_df])
    df = pd.concat([df, target_df], axis=1)
    df.to_csv(DBPlotCSVPath + 'featurePoint.csv')
    print('Done DB')

    print('Start Query')
    # Query
    p = Path(QueryPath)
    p = sorted(p.glob("*.jpg"))
    df = pd.DataFrame()
    target_df = pd.DataFrame()
    preFace = None
    for filename in tqdm(p):
        img = cv2.imread(QueryPath + filename.name)
        face, plotImg, featurePoint_df = face_shape_detector_dlib(img, filename.name)
        df = pd.concat([df, featurePoint_df])
        if len(face)==0:
            face = preFace
            print('query error')
        preFace = face
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.equalizeHist(face)
        face = cv2.resize(face,SIZE)
        cv2.imwrite(QuerySavePath + filename.name, face)
        cv2.imwrite(QueryPlotImagePath + filename.name, plotImg)
        target = filename.name[0:2]
        if target[0] == 'r':
            target = 0
        else:
            target = int(target) + 1
        t_df = pd.DataFrame({'target' : target},
                    index = [filename.name])
        target_df = pd.concat([target_df, t_df])
    df = pd.concat([df, target_df], axis=1)
    df.to_csv(QueryPlotCSVPath + 'featurePoint.csv')
    print('Done Query')

# https://qiita.com/ufoo68/items/b1379b40ae6e63ed3c79
def face_shape_detector_dlib(img, name):
    # presetting
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # frontal_face_detectorクラスは矩形, スコア, サブ検出器の結果を返す
    dets, scores, idx = detector.run(img_rgb, 0)

    if len(dets) > 0:
        for i, rect in enumerate(dets):
            if i == 0:
                shape = predictor(img_rgb, rect)
                shape = face_utils.shape_to_np(shape)
                clone = img.copy()
                min_x, max_x, min_y, max_y = 1000, 0, 1000, 0
                for (x, y) in shape:
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
                face = img[min_y:max_y, min_x:max_x]
                # cv2.putText(clone, "mouth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # landmarkを画像に書き込む
                # TODO
                # この辺をcsvに移植する作業(地獄)
                for (x, y) in shape[0:17]: # chin
                    cv2.circle(clone, (x, y), 1, (0, 0, 0), -1)
                for (x, y) in shape[17:22]: # right eyebrow
                    cv2.circle(clone, (x, y), 1, (255, 0, 0), -1)
                for (x, y) in shape[22:27]: # left eyebrow
                    cv2.circle(clone, (x, y), 1, (255, 0, 255), -1)
                for (x, y) in shape[27:36]: # nose
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                for (x, y) in shape[36:42]: # right eye
                    cv2.circle(clone, (x, y), 1, (0, 255, 255), -1)
                for (x, y) in shape[42:48]: # left eye
                    cv2.circle(clone, (x, y), 1, (255, 255, 0), -1)
                for (x, y) in shape[48:68]: # mouth
                    cv2.circle(clone, (x, y), 1, (0, 255, 0), -1)

                df = pd.DataFrame()
                for i, (x, y) in enumerate(shape[0:68]): # all
                    if i == 0:
                        part = 'chin'
                        count = 0
                    elif i == 17:
                        part = 'right_eyebrow'
                        count = 0
                    elif i == 22:
                        part = 'left_eyebrow'
                        count = 0
                    elif i == 27:
                        part = 'nose'
                        count = 0
                    elif i == 36:
                        part = 'right_eye'
                        count = 0
                    elif i == 42:
                        part = 'left_eye'
                        count = 0
                    elif i == 48:
                        part = 'mouth'
                        count = 0
                    xy_df = pd.DataFrame({
                        part + '_x' + str(count) : [x],
                        part + '_y' + str(count) : [y]},
                        index = [name])
                    df = pd.concat([df, xy_df], axis=1)
                    count += 1
                # shapeで指定した個所の切り取り画像(ROI)を取得
                # (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]])) #口の部位のみ切り出し
                # roi = img[y:y + h, x:x + w]
                # roi = cv2.resize(roi,(100,100))
        return face, clone, df
    else :
        print('error')
        return img, img, None




if __name__ == '__main__':
    cutFace()
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
        INPUT_DIR = '../input/Dlib/cutface/' + FILENAME + '/' + mode + '/csv/'
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
        df_out['right_Eyebrow_width'] = df_target_x.max(
            axis=1) - df_target_x.min(axis=1)
        df_target_y = df_target.iloc[:, 1:-1:2]
        df_out['right_Eyebrow_height'] = df_target_y.max(
            axis=1) - df_target_y.min(axis=1)

        # left eye brow(左マユ)
        df_target = df.iloc[:, leftEyebrowStart:leftEyebrowEnd]
        df_target_x = df_target.iloc[:, 0:-1:2]
        df_out['left_Eyebrow_width'] = df_target_x.max(
            axis=1) - df_target_x.min(axis=1)
        df_target_y = df_target.iloc[:, 1:-1:2]
        df_out['left_Eyebrow_height'] = df_target_y.max(
            axis=1) - df_target_y.min(axis=1)

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
        df_out['right_Eye_width'] = df_target_x.max(
            axis=1) - df_target_x.min(axis=1)
        df_target_y = df_target.iloc[:, 1:-1:2]
        df_out['right_Eye_height'] = df_target_y.max(
            axis=1) - df_target_y.min(axis=1)
        df['right_Eye_width_center'] = df_target_x.mean(axis=1)
        df['right_Eye_height_center'] = df_target_y.mean(axis=1)

        # left eye(左目)
        df_target = df.iloc[:, leftEyeStart:leftEyeEnd]
        df_target_x = df_target.iloc[:, 0:-1:2]
        df_out['left_Eye_width'] = df_target_x.max(
            axis=1) - df_target_x.min(axis=1)
        df_target_y = df_target.iloc[:, 1:-1:2]
        df_out['left_Eye_height'] = df_target_y.max(
            axis=1) - df_target_y.min(axis=1)
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
        df_out['eye2eye_dist'] = np.sqrt((df['right_Eye_width_center'] - df['left_Eye_width_center'])**2 +
                                        (df['right_Eye_height_center'] - df['left_Eye_height_center'])**2)
        df_out['Reye2nose_dist'] = np.sqrt((df['right_Eye_width_center'] - df['nose_width_center'])**2 +
                                        (df['right_Eye_height_center'] - df['nose_height_center'])**2)
        df_out['Leye2nose_dist'] = np.sqrt((df['left_Eye_width_center'] - df['nose_width_center'])**2 +
                                        (df['left_Eye_height_center'] - df['nose_height_center'])**2)
        df_out['nose2mouse_dist'] = np.sqrt((df['nose_width_center'] - df['mouse_width_center'])**2 +
                                            (df['nose_height_center'] - df['mouse_height_center'])**2)
        df_out['nose2mouse_dist'] = np.sqrt((df['nose_width_center'] - df['mouse_width_center'])**2 +
                                            (df['nose_height_center'] - df['mouse_height_center'])**2)
        df_out['Reye2mouse_dist'] = np.sqrt((df['right_Eye_width_center'] - df['mouse_width_center'])**2 +
                                            (df['right_Eye_height_center'] - df['mouse_height_center'])**2)
        df_out['Leye2mouse_dist'] = np.sqrt((df['left_Eye_width_center'] - df['mouse_width_center'])**2 +
                                            (df['left_Eye_height_center'] - df['mouse_height_center'])**2)

        for i in range(df_out.shape[0]):
            df_out.iloc[i, 1:] = df_out.iloc[i, 1:]/df_out['face_width'][i]
        df_out.drop('face_width', axis=1, inplace=True)

        df_out.to_csv(INPUT_DIR + "features_rel_dist.csv")
