# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:21:06 2018

@author: hiroki
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        
        self.defaultImg = QPixmap('../DlibDataChangeLuminace/DB/jpeg/000.jpg')
        self.changedImg = QPixmap('../DlibDataChangeLuminace/DBPlot/jpeg/000.jpg')
            
        # QPixmapオブジェクトの作成
        #pixmap = QPixmap('../DlibDataChangeLuminace/DB/jpeg/000.jpg')
        
        # ラベルを作ってその中に画像を置く
        self.lbl = QLabel()
        self.lbl.setPixmap(self.defaultImg)
        self.lbl.setAlignment(Qt.AlignCenter)
        
        # ボタン作成
        button_exec = QPushButton('実行')
        button_train = QPushButton('学習')
        button_reset = QPushButton('リセット')
        button_quit = QPushButton('終了')
        
        # スロットを作成
        button_exec.clicked.connect(self.ButtonExec)
        button_reset.clicked.connect(self.ButtonReset)
        
        # 配置
        hbox = QHBoxLayout()
        hbox.addWidget(button_exec)
        hbox.addWidget(button_train)
        hbox.addWidget(button_reset)
        
        vbox = QVBoxLayout()
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
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())