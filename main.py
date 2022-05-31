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

class Communicate(QObject):                                                 
                           
    input = pyqtSignal(list)  
    clear = pyqtSignal()                                            


class ListItem(QWidget):

    def __init__(self, itemNo, distToRoad, distToCar, rankOfDanger, parent=None):
        super(ListItem, self).__init__(parent)

        self.row = QHBoxLayout()

        self.row.addWidget(QLabel("   "+itemNo))
        self.row.addWidget(QLabel(distToCar))
        self.row.addWidget(QLabel("        "+distToRoad))
        status = QPushButton()
        if rankOfDanger==1:
            status.setIcon(QIcon("assets\\yellow.png"))
        elif rankOfDanger==2:
            status.setIcon(QIcon("assets\\red.png"))
        else:
            status.setIcon(QIcon("assets\\green.png"))
        status.setStyleSheet(" background: transparent")

        self.row.addWidget(status)

        self.setLayout(self.row)

class main(QMainWindow):
    i=False
    floor_mesh_active = False
    call=None
    listWidget = None
    inputs = None

    def __init__(self):
        super(main, self).__init__()
        self.setFixedSize(1000, 680)
        self.call=uic.loadUi('form.ui',self)
        self.listWidget = self.call.listWidget
        self.inputs = Communicate() 
        self.inputs.input.connect(self.addItemToListWidget)
        self.inputs.clear.connect(self.clearListWidget)
        
        timer = threading.Timer(0.1, self.run)
        timer.start()

    pyqtSlot()
    def clearListWidget(self):
        self.listWidget.clear()
        
    pyqtSlot(list)
    def addItemToListWidget(self, labelList):
        item = QListWidgetItem(self.listWidget)
        self.listWidget.addItem(item)
        row = ListItem(itemNo=str(labelList[0])+"  " , distToRoad=labelList[1], distToCar=labelList[2], rankOfDanger=labelList[3])
        item.setSizeHint(row.minimumSizeHint())


        self.listWidget.setItemWidget(item, row)

    def run(self):
        self.dt = Detector(self, weights='weights\\05052022_best.pt', svo=None, img_size=736, conf_thres=0.4)
        # weights\test_notebook_best.pt ev ortamında not defteri ile test yaparken kullanılabilir.

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




