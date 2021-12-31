from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from gui_designer import Ui_MainWindow
import sys
from yolo_object_dedector import *
import time

class GuiFromDesigner(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.FPSUpdate.connect(self.FPSUpdateSlot)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pBtn_close.clicked.connect(self.close)
        self.ui.pBtn_normal.clicked.connect(pBtn_normal_Clicked)
        self.ui.pBtn_gray.clicked.connect(pBtn_gray_Clicked)
        self.ui.pBtn_canny.clicked.connect(pBtn_canny_Clicked)
        self.ui.pBtn_recognition.clicked.connect(pBtn_recognition_Clicked)

    def ImageUpdateSlot(self, Image):
        self.ui.ImageViewer.setPixmap(QPixmap.fromImage(Image))

    def FPSUpdateSlot(self, value):
        self.ui.label.setText(value)

def pBtn_normal_Clicked():
    Worker1.processNumber=0

def pBtn_gray_Clicked():
    Worker1.processNumber=1

def pBtn_canny_Clicked():
    Worker1.processNumber=2

def pBtn_recognition_Clicked():
    Worker1.processNumber=3

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    FPSUpdate = pyqtSignal(str)

    processNumber = 0   

    def run(self):

        # used to record the time when we processed last frame
        prev_frame_time = 0
 
        # used to record the time at which we processed current frame
        new_frame_time = 0
        
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            while Capture.isOpened():
                ret, frame = Capture.read()
                if ret:
                    if self.processNumber == 0:
                        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        Image = cv2.flip(Image, 1)
                    elif self.processNumber == 1:
                        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        Image = cv2.cvtColor(Image, cv2.COLOR_GRAY2RGB)
                        Image = cv2.flip(Image, 1)
                    elif self.processNumber == 2:
                        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        Image = cv2.GaussianBlur(Image, (7,7), 1.5)
                        Image = cv2.Canny(Image,0,40)
                        Image = cv2.cvtColor(Image, cv2.COLOR_GRAY2RGB)
                        Image = cv2.flip(Image, 1)
                    elif self.processNumber == 3:
                        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        Image = cv2.flip(Image, 1) 
                        Image = yolo.detect(Image)

                    new_frame_time = time.time()
 
                    # Calculating the fps
 
                    # fps will be number of frame processed in given time frame
                    # since their will be most of time error of 0.001 second
                    # we will be subtracting it to get more accurate result
                    fps = 1/(new_frame_time-prev_frame_time)
                    prev_frame_time = new_frame_time
                    # converting the fps into integer
                    #fps = int(fps)
 
                    # converting the fps to string so that we can display it on frame
                    # by using putText function
                    fps = "FPS: " + str(fps)

                    ConvertTo_QtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                    Pic = ConvertTo_QtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.ImageUpdate.emit(Pic)
                    self.FPSUpdate.emit(fps)
    
    def stop(self):
        self.ThreadActive = False
        self.quit()
        
if __name__ == '__main__':
    uygulama = QApplication(sys.argv)
    pencere = GuiFromDesigner()
    yolo = YOLOv3()
    pencere.show()
    sys.exit(uygulama.exec())