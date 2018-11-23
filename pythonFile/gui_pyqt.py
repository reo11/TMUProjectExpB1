# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 20:59:12 2018

@author: hiroki
"""

import sys, os
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QHBoxLayout,
                             QVBoxLayout, QLabel, QLineEdit, QFileDialog,
                             QProgressBar)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QCoreApplication
from pathlib import Path

import numpy as np
import cv2
import pickle


class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        #self.defaultImg = QPixmap('../DlibDataChangeLuminace/DB/jpeg/000.jpg')
        self.initUI()
        
    def initUI(self):
        # ウィンドウサイズ
        self.resize(800, 500)
        self.setWindowTitle('face classification')
        
        # テキストフォルダ作成
        self.dbPath = QLineEdit()
                
        # ボタン作成
        self.trainFolder = QPushButton('trainDB参照...')
        self.btnExec = QPushButton('train実行')
        self.btnQuit = QPushButton('終了', self)
        
        # スロットを作成
        self.trainFolder.clicked.connect(lambda: self.showFolderDialog(self.dbPath))
        self.btnQuit.clicked.connect(QCoreApplication.instance().quit)
        self.btnExec.clicked.connect(self.btnExecContent)
        
        # 配置
        self.hboxTrain = QHBoxLayout()
        self.hboxTrain.addWidget(self.dbPath)
        self.hboxTrain.addWidget(self.trainFolder)
        
        self.vboxTrain = QVBoxLayout()
        self.vboxTrain.addLayout(self.hboxTrain)
        self.vboxTrain.addWidget(self.btnExec)
        self.vboxTrain.setAlignment(Qt.AlignTop)
        
        self.vboxMain = QVBoxLayout()
        self.vboxMain.addLayout(self.vboxTrain)
        self.vboxMain.addWidget(self.btnQuit)
        
        self.setLayout(self.vboxMain)
        
        self.show()

    def showFolderDialog(self, path):
        dirname = QFileDialog.getExistingDirectory(self,
                                                   'open folder',
                                                   os.path.expanduser('../'),
                                                   QFileDialog.ShowDirsOnly)
        if dirname:
            self.dirname = dirname.replace('/', os.sep)
            path.setText(self.dirname)
            
    def btnExecContent(self):
        self.showSamples()
        self.dctDB = self.dct()
        self.queryInput()
        
            
    def showSamples(self):
        # ラベルを作ってその中に画像を置く
        self.lblSamples = QHBoxLayout()
        self.lblSamples.setAlignment(Qt.AlignCenter)
        
        self.trainFileNames = self.dirFileName()
        for filename in self.trainFileNames[0:-1:20]:
            sampleImg =  QPixmap(filename).scaled(50, 50)
            lblSample = QLabel()
            lblSample.setPixmap(sampleImg)
            self.lblSamples.addWidget(lblSample)
        
        self.vboxTrain.addLayout(self.lblSamples)
        
    def dct(self):
        # プログレスバー作成
        self.dctProgress = QProgressBar(self)
        self.vboxTrain.addWidget(self.dctProgress)
        
        self.trainFileNames = self.dirFileName()
        self.compleated = 0
        
        dbList = []
        with open("dct.pickle", mode='wb') as f:
            for filename in self.trainFileNames:
                img = cv2.imread(filename, 0)
                img = cv2.Canny(img, 200, 200)
                imf = np.float32(img)/255.0
                dct = cv2.dct(imf)
                dbList.append(np.float32(dct))
                self.compleated += 100/len(self.trainFileNames)
                self.dctProgress.setValue(self.compleated)
            pickle.dump(dbList, f)
            
        return dbList
                    
    def dirFileName(self):
        p = Path(self.dbPath.text())
        p = sorted(p.glob("*.jpg"))
        filenames = []
        for name in p:
            filenames.append(str(name))
        return filenames
        
    def queryInput(self):
        self.vboxQuery = QVBoxLayout()
        self.hboxQuery = QHBoxLayout()
        # テキストフォルダ作成
        self.queryPath = QLineEdit()
        # ボタン作成
        self.queryFolder = QPushButton('Query参照...')
        self.btnInference= QPushButton('Query実行')
        # スロットを作成
        self.queryFolder.clicked.connect(lambda: self.showFolderDialog(self.queryPath))
        self.hboxQuery.addWidget(self.queryPath)
        self.hboxQuery.addWidget(self.queryFolder)
        self.vboxQuery.addLayout(self.hboxQuery)
        self.vboxQuery.addWidget(self.btnInference)
        self.vboxTrain.addLayout(self.vboxQuery)
                                
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())