# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:47:10 2019

@author: kumac
"""

import sys, os
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QHBoxLayout,
                             QVBoxLayout, QLabel, QLineEdit, QFileDialog,
                             QProgressBar, QComboBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QCoreApplication
from pathlib import Path

import numpy as np
import cv2

from processes import processes

class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        # 変数
        self.isFolder = False
        self.processNum = 0
        self.tableProcess = ["単純マッチング", "kNN", "NeuralNet", "LightGBM"]
        self.imageProcess = ["ピクセルマッチング", "CNN"]
        self.processDict = {"単純マッチング":0, "kNN":1, "NeuralNet":2, "LightGBM":3, "ピクセルマッチング":4, "CNN":5}
        self.lblEmpty = QLabel("\n", self)
        self.isChanged = False
        self.prefixAns = "<h3>Accuracy："
        
        # ウィンドウサイズ
        self.resize(500, 400)
        self.setWindowTitle('face classification')
        
        # メインの配置
        self.vboxMain = QVBoxLayout(self)
        
        # 手順1(データ選択)
        self.vboxOpe1 = QVBoxLayout(self)
        # ラベルの用意
        self.lbl1 = QLabel(self)
        self.lbl1.setText("<h2>1.処理対象のデータを選んでください")
        self.vboxOpe1.addWidget(self.lbl1)
        # combo boxでテーブルデータか画像データのフォルダを選ぶか決める
        self.intype_combo = QComboBox(self)
        self.intype_combo.addItems(["csvファイル", "画像フォルダ(.jpg)"])
        self.intype_combo.activated[str].connect(self.intypeActivated)     
        self.vboxOpe1.addWidget(self.intype_combo)
        # 入力テキストフォルダとボタンの用意
        self.hboxDB = QHBoxLayout(self)
        self.lblDB = QLabel(self)
        self.lblDB.setText("DB Path:")
        self.dbPath = QLineEdit(self)
        self.DBFolder = QPushButton('参照')
        self.DBFolder.clicked.connect(lambda: self.showFolderDialog(self.dbPath))
        self.hboxDB.addWidget(self.lblDB)
        self.hboxDB.addWidget(self.dbPath)
        self.hboxDB.addWidget(self.DBFolder)
        self.vboxOpe1.addLayout(self.hboxDB)
        
        self.hboxQuery = QHBoxLayout(self)
        self.lblQuery = QLabel(self)
        self.lblQuery.setText("Query Path:")
        self.queryPath = QLineEdit(self)
        self.queryFolder = QPushButton('参照')
        self.queryFolder.clicked.connect(lambda: self.showFolderDialog(self.queryPath))
        self.hboxQuery.addWidget(self.lblQuery)
        self.hboxQuery.addWidget(self.queryPath)
        self.hboxQuery.addWidget(self.queryFolder)
        self.vboxOpe1.addLayout(self.hboxQuery)
        
        # 手順2(処理方法選択)
        self.vboxOpe2 = QVBoxLayout(self)
        # ラベルの用意
        self.lbl2 = QLabel(self)
        self.lbl2.setText("<h2>2.処理方法を選んでください")
        self.vboxOpe2.addWidget(self.lblEmpty)
        self.vboxOpe2.addWidget(self.lbl2)
        # bombo boxで処理方法を選択
        self.process_combo = QComboBox(self)
        self.process_combo.addItems(self.tableProcess)
        self.process_combo.activated[str].connect(self.processActivated)    
        self.vboxOpe2.addWidget(self.process_combo)
        
        # 手順3(実行)
        self.vboxOpe3 = QVBoxLayout(self)
        # ラベルの用意
        self.lbl3 = QLabel(self)
        self.lbl3.setText("<h2>3.実行")
        self.vboxOpe3.addWidget(self.lblEmpty)
        self.vboxOpe3.addWidget(self.lbl3)
        # 実行ボタン
        self.btnExec = QPushButton("実行")
        self.btnExec.clicked.connect(self.btnExecContent)
        self.vboxOpe3.addWidget(self.btnExec)
        
        self.lblAccu = QLabel(self)
        self.lblAccu.setText(self.prefixAns)
        self.vboxOpe3.addWidget(self.lblAccu)
        
        
        self.vboxMain.addLayout(self.vboxOpe1)
        self.vboxMain.addLayout(self.vboxOpe2)
        self.vboxMain.addLayout(self.vboxOpe3)
        self.vboxMain.setAlignment(Qt.AlignTop)
        
        self.show()
        
        
    def showFolderDialog(self, path):
        self.isChanged = True
        if(self.isFolder):
            dirname = QFileDialog.getExistingDirectory(self,
                                                       'open folder',
                                                       os.path.expanduser('../'),
                                                       QFileDialog.ShowDirsOnly)
            if dirname:
                self.dirname = dirname.replace('/', os.sep)
                
        else:
            dirname = QFileDialog.getOpenFileName(self,
                                                  'open folder/file',
                                                  os.path.expanduser('../'))
            if dirname:
                self.dirname = dirname[0].replace('/', os.sep)
        path.setText(self.dirname)
            
        
    def intypeActivated(self, text):
        if (text == "画像フォルダ(.jpg)"):
            self.isFolder = True
            self.process_combo.clear()
            self.process_combo.addItems(self.imageProcess)
        else:
            self.isFolder = False
            self.process_combo.clear()
            self.process_combo.addItems(self.tableProcess)
      
        
    def processActivated(self, text):
        self.processNum = self.processDict[text]
            
            
    def btnExecContent(self):
        if(self.isChanged):
            self.model = processes(self.dbPath.text(), self.queryPath.text(), self.isFolder)
        accu = self.model.calc_accuracy(self.processNum)
        self.lblAccu.setText("{}{:.2%}".format(self.prefixAns, accu))
        
                                                
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())