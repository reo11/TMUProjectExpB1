# -*- coding: utf-8 -*-
import cv2
import os
import dlib
import imutils
import numpy as np
from pathlib import Path
from imutils import face_utils

DBPath = '../dataset/DB/jpeg/'
QueryPath = '../dataset/Query/jpeg/'
SIZE = (200, 200)

FILENAME = 'DlibData'
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
    for filename in p:
        img = cv2.imread(DBPath + filename.name)
        ### frame is arranged image
        ### roi is a part of face
        frame, roi = face_shape_detector_dlib(img)
        cv2.imshow('img', frame)
        #cv2.imshow('img', roi)
        cv2.waitKey(100)
    print('Done DB')
    print('Start Query')
    # Query
    p = Path(QueryPath)
    p = sorted(p.glob("*.jpg"))
    for filename in p:
        img = cv2.imread(QueryPath + filename.name)
        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            face = preProcessing(face)
        cv2.imwrite(QuerySavePath + filename.name,face)
    print('Done Query')

def preProcessing(img):
    img = cv2.resize(img, SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # #Sobelフィルタでx方向のエッジ検出
    # gray_sobelx = cv2.Sobel(img,cv2.CV_32F,1,0)
    # #Sobelフィルタでy方向のエッジ検出
    # gray_sobely = cv2.Sobel(img,cv2.CV_32F,0,1)
    # #8ビット符号なし整数変換
    # gray_abs_sobelx = cv2.convertScaleAbs(gray_sobelx)
    # gray_abs_sobely = cv2.convertScaleAbs(gray_sobely)
    # #重み付き和
    # img = cv2.addWeighted(gray_abs_sobelx,0.5,gray_abs_sobely,0.5,0)
    # img = (img - np.mean(img))/np.std(img)*32+100
    img = cv2.equalizeHist(img)
    # img = cv2.Canny(img,100,200)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

# https://qiita.com/ufoo68/items/b1379b40ae6e63ed3c79
def face_shape_detector_dlib(img):
    # presetting
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    # frontal_face_detectorクラスは矩形, スコア, サブ検出器の結果を返す
    dets, scores, idx = detector.run(img_rgb, 0)
    if len(dets) > 0:
        for i, rect in enumerate(dets):
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
            # shapeで指定した個所の切り取り画像(ROI)を取得
            (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]])) #口の部位のみ切り出し
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi,(100,100))
        return clone, roi
    else :
        return img, None

if __name__ == '__main__':
    cutFace()