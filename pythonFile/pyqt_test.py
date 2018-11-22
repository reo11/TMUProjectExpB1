# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:21:06 2018

@author: hiroki
"""

import sys, os
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QHBoxLayout,
                             QVBoxLayout, QLabel, QLineEdit, QFileDialog)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        self.defaultImg = QPixmap('../DlibDataChangeLuminace/DB/jpeg/000.jpg')
        self.changedImg = QPixmap('../DlibDataChangeLuminace/DBPlot/jpeg/000.jpg')
        self.initUI()
        
    def initUI(self):
            
        # QPixmapオブジェクトの作成
        #pixmap = QPixmap('../DlibDataChangeLuminace/DB/jpeg/000.jpg')
        
        # テキストフォルダ作成
        self.txtFolder = QLineEdit()
        
        # ラベルを作ってその中に画像を置く
        self.lbl = QLabel()
        self.lbl.setPixmap(self.defaultImg)
        self.lbl.setAlignment(Qt.AlignCenter)
        
        # ボタン作成
        button_exec = QPushButton('実行')
        button_train = QPushButton('学習')
        button_reset = QPushButton('リセット')
        button_quit = QPushButton('終了')
        self.btnFolder = QPushButton('参照...')
        
        # スロットを作成
        button_exec.clicked.connect(self.ButtonExec)
        button_reset.clicked.connect(self.ButtonReset)
        self.btnFolder.clicked.connect(self.showFolderDialog)
        
        # 配置
        self.hboxText = QHBoxLayout()
        self.hboxText.addWidget(self.txtFolder)
        self.hboxText.addWidget(self.btnFolder)
        
        hbox = QHBoxLayout()
        hbox.addWidget(button_exec)
        hbox.addWidget(button_train)
        hbox.addWidget(button_reset)
        
        vbox = QVBoxLayout()
        vbox.addLayout(self.hboxText)
        vbox.addWidget(self.lbl)
        vbox.addLayout(hbox)
        vbox.addWidget(button_quit)
        
        self.setLayout(vbox)
        
        self.show()
        
    def ButtonExec(self):
        # QPixmapオブジェクトの作成
        #pixmap = QPixmap('../DlibDataChangeLuminace/DBPlot/jpeg/000.jpg')
        self.lbl.setPixmap(self.changedImg)
        
    def ButtonReset(self):
        # QPixmapオブジェクトの作成
        self.lbl.setPixmap(self.defaultImg)
        
    def showFolderDialog(self):
        dirname = QFileDialog.getExistingDirectory(self,
                                                   'open folder',
                                                   os.path.expanduser('.'),
                                                   QFileDialog.ShowDirsOnly)
        if dirname:
            self.dirname = dirname.replace('/', os.sep)
            self.txtFolder.setText(self.dirname)
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())