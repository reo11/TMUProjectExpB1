import numpy as np
import cv2
from scipy.fftpack import dct
from pathlib import Path
from time import sleep
from termcolor import colored
from tqdm import tqdm


FILENAME = 'dataset2'

dbPath = '../' + FILENAME + '/DB/jpeg/'
queryPath = '../' + FILENAME + '/Query/jpeg/'
DBCannySavePath = '../dct/DB/canny/jpeg/'
QueryCannySavePath = '../dct/Query/canny/jpeg/'
DBDctSavePath = '../dct/DB/dctAfterCanny/jpeg/'
QueryDctSavePath = '../dct/Query/dctAfterCanny/jpeg/'
DBDctOriginalSavePath = '../dct/DB/dctOriginal/jpeg/'
QueryDctOriginalSavePath = '../dct/Query/dctOriginal/jpeg/'
best_score = 0

if __name__ == '__main__':
    for i in tqdm(range(10,40)):
    #with open("dct.csv", mode='w') as f:
        for j in range(i,i*3):
            dbList = []
            queryList = []
            distanceList = []
            answerList = []
            count = 0
            # print('c1: ' + str(i) + ', c2: ' + str(j))
            #f.write('query_name,answer\n')
            p = Path(dbPath)
            p = sorted(p.glob("*.jpg"))
            for filename in p:
                img = cv2.imread(dbPath + filename.name, 0)
                # cv2.imwrite(DBDctOriginalSavePath + filename.name, cv2.dct(np.float32(img)))
                img = cv2.GaussianBlur(img, (7, 7), 2)
                img = cv2.Canny(img, i, j)
                # img = cv2.Canny(img, c1, c2)
                # cv2.imwrite(DBCannySavePath + filename.name, img)
                imf = np.float32(img)
                dst = cv2.dct(imf)
                # cv2.imwrite(DBDctSavePath + filename.name, dst)
                dbList.append(np.float32(dst))
            p = Path(queryPath)
            p = sorted(p.glob("*.jpg"))
            for filename in p:
                distanceList = []
                img = cv2.imread(queryPath + filename.name, 0)
                # cv2.imwrite(QueryDctOriginalSavePath + filename.name, cv2.dct(np.float32(img)))
                img = cv2.GaussianBlur(img, (7, 7), 2)
                img = cv2.Canny(img, i, j)
                # img = cv2.Canny(img, c1, c2)
                # cv2.imwrite(QueryCannySavePath + filename.name, img)
                imf = np.float32(img)
                dst = cv2.dct(imf)
                # cv2.imwrite(QueryDctSavePath + filename.name, dst)
                queryList.append(np.float32(dst))
                for index in range(len(dbList)):
                    distance  = (dbList[index][:15, :15] - queryList[-1][:15, :15])**2
                    distanceList.append(distance.sum())
                distSort = np.argsort(distanceList)
                answerList.append(np.uint8(np.argmin(distanceList)/10)+1)
                ranking = np.uint8(distSort[:10]/10)+1
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
                    #print(colored(filename.name + ', mode: ' + str(mode) + ', answer: ' + str(answerList[-1]), 'blue'))
                if target == answerList[-1]:
                    count += 1
                #     print(colored(filename.name + ', mode: ' + str(mode) + ', answer: ' + str(answerList[-1]) + ', correct answer: ' + str(target), 'blue'))
                # else:
                #     print(colored(filename.name + ', mode: ' + str(mode) + ', answer: ' + str(answerList[-1]) + ', correct answer: ' + str(target), 'red'))
                # distanceList = []
                # f.write(filename.name + ',' + str(answerList[-1]) + '\n')
            accuracy  = count / len(queryList) * 100
            if best_score < accuracy:
                best_score = accuracy
                c1 = i
                c2 = j
                print('change Score: ' + str(int(best_score)) + 'c1: ' + str(c1) + ', c2: ' + str(c2))
            #print(str(accuracy) + '%')
    print(str(best_score) + '%')
    print('c1: ' + str(c1) + ', c2: ' + str(c2))
