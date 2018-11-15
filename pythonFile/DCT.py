import numpy as np
import cv2
from scipy.fftpack import dct
from pathlib import Path
from time import sleep

FILENAME = 'facedata'

dbPath = '../' + FILENAME + '/DB/jpeg/'
queryPath = '../' + FILENAME + '/Query/jpeg/'

#filename = DBPath + '000.jpg'

if __name__ == '__main__':
    dbList = []
    queryList = []
    distanceList = []
    answerList = []
    p = Path(dbPath)
    p = sorted(p.glob("*.jpg"))
    for filename in p:
        img = cv2.imread(dbPath + filename.name, 0)
        img = cv2.Canny(img, 200, 200)
        imf = np.float32(img)/255.0
        dst = cv2.dct(imf)
        dbList.append(np.float32(dst))

    sleep(1)
    p = Path(queryPath)
    p = sorted(p.glob("*.jpg"))
    for filename in p:
        img = cv2.imread(queryPath + filename.name, 0)
        img = cv2.Canny(img, 200, 200)
        imf = np.float32(img)/255.0
        dst = cv2.dct(imf)
        queryList.append(np.float32(dst))
        for index in range(len(dbList)):
            distance  = (dbList[index][:15, :15] - queryList[-1][:15, :15])**2
            distanceList.append(distance.sum())
        answerList.append(np.uint8(np.argmin(distanceList)/10))
        distanceList = []
    np.savetxt("dct.csv", answerList, delimiter=",")

