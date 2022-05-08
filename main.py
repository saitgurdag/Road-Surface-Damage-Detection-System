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
    # create two new signals on the fly: one will handle                    
    # int type, the other will handle strings                               
    input = pyqtSignal(list)  
    clear = pyqtSignal()                                            


class ListItem(QWidget):

    def __init__(self, itemNo, distToRoad, distToCar, rankOfDanger, parent=None):
        super(ListItem, self).__init__(parent)

        self.row = QHBoxLayout()

        self.row.addWidget(QLabel("  "+itemNo))
        self.row.addWidget(QLabel(distToCar))
        self.row.addWidget(QLabel(distToRoad))
        status = QPushButton()
        if rankOfDanger==1:
            status.setIcon(QIcon("assets\\yellow.png"))
        elif rankOfDanger==2:
            status.setIcon(QIcon("assets\\red.png"))
        else:
            status.setIcon(QIcon("assets\\green.png"))
        status.setStyleSheet(" background: transparent")
        # pixmap = QPixmap("assets\yellow.png")
        # qLabel.setPixmap(pixmap)
        # qLabel.setMask(pixmap.mask())
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
        self.call.meshOnOff.setIcon(QIcon("assets\off-button.png"))
        self.call.meshOnOff.setStyleSheet(" background: transparent")
        self.call.meshOnOff.setFixedSize(QSize(100,30))
        self.call.meshOnOff.setIconSize(QSize(48,48))
        self.call.meshOnOff.clicked.connect(self.meshOnOffClicked)
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

        # Associate the custom widget to the list entry
        self.listWidget.setItemWidget(item, row)

    def meshOnOffClicked(self):
        self.floor_mesh_active = not self.floor_mesh_active
        if self.floor_mesh_active:
            self.call.meshOnOff.setIcon(QIcon("assets\on-button.png"))
        else:
            self.call.meshOnOff.setIcon(QIcon("assets\off-button.png"))
        print(self.floor_mesh_active)

    def run(self):
        self.dt = Detector(self, weights='weights\\05052022_best.pt', svo=None, img_size=1280, conf_thres=0.2 )

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




