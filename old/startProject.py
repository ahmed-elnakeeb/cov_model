import os
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUi
from PyQt5 import uic
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap, QImage
import subprocess
import cv2
import numpy as np           
import pandas as pd


qtCreatorFile = "uidesign.ui"  # Enter file here.
global ImageFile
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent=parent)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.browse.clicked.connect(self.Test)
        self.classifierBtn.clicked.connect(self.runClassifier)
        self.close.clicked.connect(self.Close)
        self.comboBox.addItem("covid")
        self.comboBox.addItem("penomena")
        self.comboBox.addItem("sars")
        self.comboBox.addItem("Tuberculosis")

    # Convert an opencv image to QPixmap
    def drawImage(self, cvImg, imgTitle):
        self.labelImg_title.setText(imgTitle)   # Title of the current viewed image
        # Convert image to QImage
        scene = QtWidgets.QGraphicsScene()
        pixmap = QImage(cvImg.data, cvImg.shape[1], cvImg.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap(pixmap)
        scene.addPixmap(pix)
        self.graphicsView.setScene(scene)   # set the current viewed image
    
    def Test(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        ImageFile = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image To Process", "","All Files (*);;Image Files(*.jpg *.gif)", options=options)
        if ImageFile:
            self.filename_text.setText(ImageFile[0])
        exec(open("imageinfo.py").read())
        self.classifierBtn.setEnabled(True)
    
    def runClassifier(self):
        #import classifier
        exec(open('detector.py').read())

    def Close(self):
        self.destroy()
        QCoreApplication.instance().quit
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())