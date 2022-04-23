from detector import Detector

from PyQt5.QtCore import pyqtSlot, QObject

class HomeFunctions(QObject):

    def __init__(self) :
        dt = Detector(weights='weights\\yeni.pt', svo=None, img_size=416, conf_thres=0.4)

    @pyqtSlot()
    def changePage(self):
        print("Button was pressed")