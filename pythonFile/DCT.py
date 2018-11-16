import numpy as np
import cv2
from scipy.fftpack import dct
from pathlib import Path
from time import sleep
from termcolor import colored

FILENAME = 'facedata'

dbPath = '../' + FILENAME + '/DB/jpeg/'
queryPath = '../' + FILENAME + '/Query/jpeg/'

#filename = DBPath + '000.jpg'

if __name__ == '__main__':
    dbList = []
    queryList = []
    distanceList = []
    answerList = []
    count = 0

    with open("dct.csv", mode='w') as f:
        f.write('query_name,answer\n')
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
            distSort = np.argsort(distanceList)
            answerList.append(np.uint8(np.argmin(distanceList)/10)+1)
            ranking = np.uint8(distSort[:20]/10)+1
            mode = np.argmax(np.bincount(ranking))
            #print(str(ranking) + '   ' + str(answerList[-1]))
            #print('mode: ' + str(mode))
            # if mode != answerList[-1]:
            target = filename.name[0:2]
            if target[0] == 'r':
                target = 0
            else:
                target = int(target) + 1
            # 結果に応じて色を付ける
            # if target == mode :
            #     print(colored(filename.name + ', mode: ' + str(mode) + ', answer: ' + str(answerList[-1]), 'blue'))
            if target == answerList[-1]:
                count += 1
                print(colored(filename.name + ', mode: ' + str(mode) + ', answer: ' + str(answerList[-1]) + ', correct answer: ' + str(target), 'blue'))
            else:
                print(colored(filename.name + ', mode: ' + str(mode) + ', answer: ' + str(answerList[-1]) + ', correct answer: ' + str(target), 'red'))
            distanceList = []
            f.write(filename.name + ',' + str(answerList[-1]) + '\n')
        accuracy  = count / len(queryList) * 100    
        print(str(accuracy) + '%')
