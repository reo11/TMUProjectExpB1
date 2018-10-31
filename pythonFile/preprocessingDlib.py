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

DBPath = '../dataset/DB/jpeg/'
QueryPath = '../dataset/Query/jpeg/'
SIZE = (200, 200)

FILENAME = 'DlibDataChangeLuminace'

DBSavePath = '../' + FILENAME + '/DB/jpeg/'
QuerySavePath = '../' + FILENAME + '/Query/jpeg/'
DBPlotImagePath = '../' + FILENAME + '/DBPlot/jpeg/'
QueryPlotImagePath = '../' + FILENAME + '/QueryPlot/jpeg/'
DBPlotCSVPath = '../' + FILENAME + '/DB/csv/'
QueryPlotCSVPath = '../' + FILENAME + '/Query/csv/'

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
    for index, filename in enumerate(p):
        img = cv2.imread(DBPath + filename.name)
        face, plotImg, featurePoint_df = face_shape_detector_dlib(img, filename.name)
        df = pd.concat([df, featurePoint_df])
        if len(face)==0:
            face = preFace
        preFace = face
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.equalizeHist(face)
        cv2.imwrite(DBSavePath + filename.name, face)
        cv2.imwrite(DBPlotImagePath + filename.name, plotImg)
        #print(featurePoint_df)
        target = int(index / 10) + 1
        t_df = pd.DataFrame({'target' : target},
                    index = [filename.name])
        target_df = pd.concat([target_df, t_df])
        print(target_df)
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
    for filename in p:
        img = cv2.imread(QueryPath + filename.name)
        face, plotImg, featurePoint_df = face_shape_detector_dlib(img, filename.name)
        df = pd.concat([df, featurePoint_df])
        if len(face)==0:
            face = preFace
        preFace = face
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.equalizeHist(face)
        cv2.imwrite(QuerySavePath + filename.name, face)
        cv2.imwrite(QueryPlotImagePath + filename.name, plotImg)
        target = filename.name[0:2]
        if target[0] == 'r':
            target = 0
        else:
            target = int(target)
        t_df = pd.DataFrame({'target' : target},
                    index = [filename.name])
        target_df = pd.concat([target_df, t_df])
        print(featurePoint_df)
    df = pd.concat([df, target_df], axis=1)
    df.to_csv(QueryPlotCSVPath + 'featurePoint.csv')
    print('Done Query')

# https://qiita.com/ufoo68/items/b1379b40ae6e63ed3c79
def face_shape_detector_dlib(img, name):
    # presetting
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    # frontal_face_detectorクラスは矩形, スコア, サブ検出器の結果を返す
    dets, scores, idx = detector.run(img_rgb, 0)
    if len(dets) > 0:
        for i, rect in enumerate(dets):
            x = rect.left()
            y = rect.top()
            w = rect.width()
            h = rect.height()
            face = img[y:y+h, x:x+w]

            shape = predictor(img_rgb, rect)
            shape = face_utils.shape_to_np(shape)
            clone = img.copy()

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
        return img, img, None

if __name__ == '__main__':
    cutFace()