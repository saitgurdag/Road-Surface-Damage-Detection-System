import os
from pathlib import Path
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
from PyQt5 import uic
from PyQt5 import QtCore

import threading
from detector import Detector

class main(QMainWindow):
    i=False
    floor_mesh_active = False

    def __init__(self):
        super(main, self).__init__()
        self.setFixedSize(1024, 600)
        call=uic.loadUi('form.ui',self)
        call.meshOnOff.clicked.connect(self.meshOnOffClicked)
        timer = threading.Timer(0.1, self.run)
        timer.start()


    def meshOnOffClicked(self):
        self.floor_mesh_active = not self.floor_mesh_active
        print(self.floor_mesh_active)

    def run(self):
        self.dt = Detector(self, weights='weights\\04042022_best.pt', svo=None, img_size=416, conf_thres=0.4)

    def displayImage(self,lbl, img,window=1):
        qformat=QImage.Format_Indexed8

        if len(img.shape)==3:
            if(img.shape[2])==4:
                qformat=QImage.Format_RGBA8888

            else:
                qformat=QImage.Format_RGB888

        img=QImage(img,img.shape[1],img.shape[0],qformat)
        img=img.rgbSwapped()
        lbl.setPixmap(QPixmap.fromImage(img))
        lbl.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = main()
    widget.show()
    sys.exit(app.exec_())

