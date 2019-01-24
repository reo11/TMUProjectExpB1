import numpy as np
import pandas as pd
import cv2
from scipy.fftpack import dct
from pathlib import Path
from time import sleep
from termcolor import colored
from tqdm import tqdm

hist = 'default'
FILENAME = '../input/Dlib/cutface/'

dbPath = FILENAME + hist + '/DB/jpeg/'
queryPath = FILENAME + hist + '/Query/jpeg/'
DBCannySavePath = '../dct/DB/canny/jpeg/'
QueryCannySavePath = '../dct/Query/canny/jpeg/'
DBDctSavePath = '../dct/DB/dctAfterCanny/jpeg/'
QueryDctSavePath = '../dct/Query/dctAfterCanny/jpeg/'
DBDctOriginalSavePath = '../dct/DB/dctOriginal/jpeg/'
QueryDctOriginalSavePath = '../dct/Query/dctOriginal/jpeg/'
best_score = 0

if __name__ == '__main__':
    #with open("dct.csv", mode='w') as f:
    dbList = []
    queryList = []
    distanceList = []
    answerList = []
    count = 0
    bestAcc = 0

    p = Path(dbPath)
    p = sorted(p.glob("*.jpg"))
    feature_name_list = []
    for i in range(225):
        featrue_name = 'dct_' + str(i)
        feature_name_list.append(featrue_name)
    db_df = pd.DataFrame(columns=feature_name_list)
    for index, filename in enumerate(tqdm(p)):
        img = cv2.imread(dbPath + filename.name, 0)

        # img = cv2.GaussianBlur(img, (7, 7), 2)
        # img = cv2.Canny(img, 13, 39)

        imf = np.float32(img)
        dst = cv2.dct(imf)
        dst = dst[:15,:15]
        for ind, name in enumerate(feature_name_list):
            db_df.loc[index, name] = np.array(dst).flatten()[ind]

        dbList.append(np.float32(dst))
    p = Path(queryPath)
    p = sorted(p.glob("*.jpg"))
    query_df = pd.DataFrame(columns=feature_name_list)
    for index, filename in enumerate(tqdm(p)):
        distanceList = []
        img = cv2.imread(queryPath + filename.name, 0)

        # img = cv2.GaussianBlur(img, (7, 7), 2)
        # img = cv2.Canny(img, 13, 39)

        imf = np.float32(img)
        dst = cv2.dct(imf)
        dst = dst[:15,:15]
        for ind, name in enumerate(feature_name_list):
            query_df.loc[index, name] = np.array(dst).flatten()[ind]

        queryList.append(np.float32(dst))
        for index in range(len(dbList)):
            distance  = (dbList[index][:15, :15] - queryList[-1][:15, :15])**2
            distanceList.append(distance.sum())
        distSort = np.argsort(distanceList)
        answerList.append(np.uint8(np.argmin(distanceList)/10))
        ranking = np.uint8(distSort[:10]/10)
        mode = np.argmax(np.bincount(ranking))

        target = filename.name[0:2]
        if target[0] == 'r':
            target = 20
        else:
            target = int(target)
        # 結果に応じて色を付ける
        if target == mode :
            print(colored(filename.name + ', mode: ' + str(mode) + ', answer: ' + str(answerList[-1]), 'blue'))
        if target == answerList[-1]:
            count += 1
            print(colored(filename.name + ', mode: ' + str(mode) + ', answer: ' + str(answerList[-1]) + ', correct answer: ' + str(target), 'blue'))
        else:
            print(colored(filename.name + ', mode: ' + str(mode) + ', answer: ' + str(answerList[-1]) + ', correct answer: ' + str(target), 'red'))
        distanceList = []

    if db_df.min().min() < query_df.min().min():
        norm_min = db_df.min().min()
    else:
        norm_min = query_df.min().min()

    db_df = db_df - norm_min
    query_df = query_df - norm_min

    if db_df.max().max() > query_df.max().max():
        norm_max = db_df.max().max()
    else:
        norm_max = query_df.max().max()

    db_df = db_df / norm_max
    query_df = query_df / norm_max

    p = Path(dbPath)
    p = sorted(p.glob("*.jpg"))
    for index, filename in enumerate(tqdm(p)):
        db_df.loc[index, 'target'] = int(index/10)
    db_df.to_csv("../input/dct/noCanny/" + hist + "/db_list.csv")

    p = Path(queryPath)
    p = sorted(p.glob("*.jpg"))
    for index, filename in enumerate(tqdm(p)):
        target = filename.name[0:2]
        if target[0] == 'r':
            target = 0
        else:
            target = int(target)
        query_df.loc[index, 'target'] = target
    query_df.to_csv("../input/dct/noCanny/" + hist + "/query_list.csv")

    accuracy  = count / len(queryList) * 100
    print(str(accuracy) + '%')