# -*- coding: utf-8 -*-
import cv2
from pathlib import Path

DBPath = '../dataset/DB/jpeg/'
QueryPath = '../dataset/Query/jpeg/'
SIZE = (200, 200)

def cutFace(type):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_' + type + '.xml')
    DBSavePath = '../' + type + '/DB/jpeg/'
    QuerySavePath = '../' + type + '/Query/jpeg/'
    # Haar-like特徴分類器の読み込み

    print('Start DB')
    # Database
    p = Path(DBPath)
    p = sorted(p.glob("*.jpg"))
    count = 0
    for filename in p:
        img = cv2.imread(DBPath + filename.name)
        faces = face_cascade.detectMultiScale(img)
        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, SIZE)
        cv2.imwrite(DBSavePath + filename.name,face)
        count += 1
    print('Done DB')
    print('Start Query')
    # Query
    p = Path(QueryPath)
    p = sorted(p.glob("*.jpg"))
    count = 0
    for filename in p:
        img = cv2.imread(QueryPath + filename.name)
        faces = face_cascade.detectMultiScale(img)
        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, SIZE)
        cv2.imwrite(QuerySavePath + filename.name,face)
        count += 1
    print('Done Query')

# cutFace('frontalface_default')
# cutFace('frontalface_alt')
# cutFace('frontalface_alt2')

cutFace('frontalface_alt_tree')