# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

DBPath = '../input/default/dataset/DB/jpeg/'
QueryPath = '../input/default/dataset/Query/jpeg/'
SIZE = (128, 128)
# face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_' + type + '.xml')
face_cascade_frontalface_default = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
face_cascade_frontalface_alt = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
face_cascade_frontalface_alt2 = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
face_cascade_frontalface_alt_tree = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt_tree.xml')

def cutFace(type):
    DBSavePath = '../input/opencv/' + type + '/DB/jpeg/'
    QuerySavePath = '../input/opencv/' + type + '/Query/jpeg/'

    # Haar-like特徴分類器の読み込み

    print('Start DB')
    # Database
    p = Path(DBPath)
    p = sorted(p.glob("*.jpg"))
    count = 0
    for filename in tqdm(p):
        img = cv2.imread(DBPath + filename.name)
        faces = findFace(filename.name, img)
        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            face = preProcessing(face)
        cv2.imwrite(DBSavePath + filename.name,face)
        count += 1
    print('Done DB')
    print('Start Query')
    # Query
    p = Path(QueryPath)
    p = sorted(p.glob("*.jpg"))
    count = 0
    for filename in tqdm(p):
        img = cv2.imread(QueryPath + filename.name)
        faces = findFace(filename.name, img)
        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            face = preProcessing(face)
        cv2.imwrite(QuerySavePath + filename.name,face)
        count += 1
    print('Done Query')

def findFace(name, img):
    faces = face_cascade_frontalface_alt_tree.detectMultiScale(img)
    if len(faces)==0:
        faces = face_cascade_frontalface_alt.detectMultiScale(img)
        if len(faces)==0:
            faces = face_cascade_frontalface_alt2.detectMultiScale(img)
            if len(faces)==0:
                faces = face_cascade_frontalface_default.detectMultiScale(img)
                if len(faces)==0:
                    print(name + ' not found')
    return faces

def preProcessing(img):
    img = cv2.resize(img, SIZE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    # img = cv2.equalizeHist(img)
    # img = cv2.Canny(img,100,200)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

# cutFace('frontalface_default')
# cutFace('frontalface_alt')
# cutFace('frontalface_alt2')

cutFace('cutface')
