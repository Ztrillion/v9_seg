from PySide6 import QtCore, QtWidgets, QtGui
import cv2, os
from ultralytics import YOLO
import numpy as np
import sys
from collections import defaultdict
class Colors:
    def __init__(self):
        hexs = (
            "FF3838","FF9D97","FF701F","FFB21D","CFD231","48F90A","92CC17","3DDB86","1A9334","00D4BB",
            "2C99A8","00C2FF","344593","6473FF","0018EC","8438FF","520085","CB38FF","FF95C8","FF37C7",)
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

class ImageViewer(QtWidgets.QMainWindow):
    qss = """
        QMainWindow {
        background-color: #000000;
        border: 2px solid #2EFEF7;
        border-radius: 5px;
        }

        QLabel#videoLabel{
        background-color : transparent; 
        border: 3px solid; 
        border-radius:10px; 
        border-color : #2EFEF7;   
        color:white; 
        font-size :15; 
        font-weight : bold;
        }

        QMessageBox QLabel{background-color : transparent; color:black; font-size :40; font-weight : bold;}
        QMessageBox QPushButton{background-color : transparent; color:black; font-size :40; font-weight : bold;}
        
        QRadioButton {
        border: 3px solid; border-radius:10px; color:#2EFEF7; 
        font : 10pt Courier New; font-weight : bold;
        }

        QMenuBar {
        background-color : transparent; border: 3px solid; border-radius:5px; border-color : #2EFEF7; font : 13pt Courier New; color : white; font-weight : bold; text-align : center;
        }
        
        QPushButton {
        background-color : transparent; border: 3px solid; border-radius:5px; border-color : #2EFEF7; font : 13pt Courier New; color : white; font-weight : bold; text-align : center;
        }

        QListWidget {background-color : transparent; border: 3px solid; border-radius:10px; border-color : #2EFEF7; color:white; font-size :15px; font-weight : bold;}
        QListWidget::item:selected{background: #FFB0C4DE;}
        
    """ 
    def __init__(self,parent=None):
        super(ImageViewer, self).__init__(parent)
        QtWidgets.QApplication.processEvents()
        self.initialize_ui_elements()
        self.setCentralWidget(self.centralWidget)
        self.show()
        
        
    def initialize_ui_elements(self):
        self.setWindowIcon(QtGui.QIcon('./logo/combus_logo2.png'))
        self.resize(1500, 800)
        self.centralWidget = QtWidgets.QWidget()
        self.setWindowTitle("IMAGE VIEWER")
        MainQvBox = QtWidgets.QVBoxLayout()
        MainGrid = QtWidgets.QGridLayout()
        
        self.videoLabel = QtWidgets.QLabel(self)
        self.videoLabel.setScaledContents(True)
        self.videoLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.videoLabel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,QtWidgets.QSizePolicy.Policy.Ignored)
        self.videoLabel.setObjectName("videoLabel")
        
        self.detected_objects_list = QtWidgets.QListWidget(self)
        self.detected_objects_list.setObjectName("detected_objects_list")
        self.detected_objects_list.itemSelectionChanged.connect(self.highlight_selected)
        self.detected_objects_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        self.basic_mode_checkbox = QtWidgets.QRadioButton(self)
        self.basic_mode_checkbox.setText("WEBCAM MODE")
        self.basic_mode_checkbox.clicked.connect(self.webcam)

        self.LABELED_checkbox = QtWidgets.QRadioButton(self)
        self.LABELED_checkbox.clicked.connect(self.loadVideo)
        self.LABELED_checkbox.setText("VIDEO PLAYER MODE")
        

        fileload_Action = QtGui.QAction('Video load', self)
        fileload_Action.setShortcut('Ctrl+F')
        fileload_Action.setStatusTip('비디오 파일 로드')
        fileload_Action.triggered.connect(self.loadVideo)

        
        menubar = QtWidgets.QMenuBar(self)
        self.file_menu = menubar.addMenu('OPTION')
        self.file_menu.addAction(fileload_Action)


        mode_qhbox = QtWidgets.QHBoxLayout()
        mode_qhbox.addWidget(self.basic_mode_checkbox)
        mode_qhbox.addWidget(self.LABELED_checkbox)

        MainGrid.addWidget(menubar,0,0,1,1)
        MainGrid.addLayout(mode_qhbox,0,10,1,11)
        MainGrid.addWidget(self.videoLabel, 2,0,15,16)
        MainGrid.addWidget(self.detected_objects_list, 2, 16, 15, 5)

        MainQvBox.addLayout(MainGrid)
        self.centralWidget.setLayout(MainQvBox)
        self.setStyleSheet(self.qss)
        self.setCentralWidget(self.centralWidget)
        self.general_model = YOLO("lib/yolov8m-seg.pt")
        self.timer = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.playVideo)
        self.selected_track_id = None
        self.track_history = defaultdict(lambda: [])
        self.colors = Colors()

    '''
    -------------------------------------- DETECT IMAGE, DE-IDENTIFICATION FUNCTION ----------------------------------------------
    '''


    def webcam(self):
        
        try :
            self.cap = cv2.VideoCapture(0)
            self.timer.start(10) 
            #self.playVideo()
            
        except :
            self.Information_message( f"Load model", f"Please connect your webcam")

    def loadVideo(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Video', '', 'Video Files (*.mp4 *.avi)')
        if filePath != '':
            self.cap = cv2.VideoCapture(os.path.abspath(filePath)) 
            self.timer.start(30)


    def playVideo(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame,1)
            h, w, ch = frame.shape
            self.show_detected_video(frame,h,w,ch)

    # model로 image를 예측하고 예측된 객체 정보 self.detected_objects_list에 추가    
    def show_detected_video(self, frame,h, w, ch):
        
        results=self.general_model.track(frame, persist=True, show=False,tracker='botsort.yaml',conf=0.4)
        current_track_ids = set()
        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls_ids = results[0].boxes.cls
            
            for mask, track_id,box,cls in zip(masks, track_ids, boxes,cls_ids):
                if self.general_model.names[int(cls)] == 'person':
                    track_label = f"{self.general_model.names[int(cls)].upper()}_Tracking_ID = {track_id}"
                    track = self.track_history[track_id]
                    track.append((float(box[0]), float(box[1])))
                    if len(track) > 200: 
                        track.pop(0)
                    
                    current_track_ids.add(track_label)
                    self.detect_draw_masks_tracking(frame=frame, mask=mask,track=track,mask_color=self.colors(track_id, True),track_label=f"{[track_id]}{self.general_model.names[int(cls)]}")
                    if not self.detected_objects_list.findItems(track_label, QtCore.Qt.MatchExactly):
                        self.detected_objects_list.addItem(track_label)
                else : 
                    label = f"{self.general_model.names[int(cls)].upper()}"
                    self.detect_draw_masks(frame=frame, mask=mask,mask_color=self.colors(track_id, True),track_label=f"{label}")
                    if not self.detected_objects_list.findItems(label, QtCore.Qt.MatchExactly):
                        self.detected_objects_list.addItem(label)
                        
        for i in range(self.detected_objects_list.count()-1, -1, -1):
            item = self.detected_objects_list.item(i)
            if item.text() not in current_track_ids:
                self.detected_objects_list.takeItem(i)
        self.display_video(frame,h, w, ch)    
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('output.avi', fourcc, fps, (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        out.write(frame)
        return frame
    
    def detect_draw_masks_tracking(self, frame, mask, track,mask_color=(255, 0, 255), track_label=None):
        
        cv2.polylines(frame, [np.int32([mask])], isClosed=True, color=mask_color, thickness=1)
        label = f"{track_label}" 
        text_size, _ = cv2.getTextSize(label, 0, 0.7, 1)
        
        # cv2.rectangle(
        #     frame,
        #     (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10),
        #     (int(mask[0][0]) + text_size[0] // 2 + 5, int(mask[0][1] + 5)),
        #     mask_color,
        #     -1,
        # )

        cv2.putText(
            frame, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1]) - 5), 0, 0.7, mask_color, 2
        )  
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False,color=(37, 255, 225), thickness=2)
        cv2.circle(frame,(int(track[-1][0]), int(track[-1][1])),5, (235, 219, 11), -1)   
        return frame  

    def detect_draw_masks(self, frame, mask,mask_color=(255, 0, 255), track_label=None):
        cv2.polylines(frame, [np.int32([mask])], isClosed=True, color=mask_color, thickness=1)
        label = f"{track_label}" 
        text_size, _ = cv2.getTextSize(label, 0, 0.7, 1)
        
        # cv2.rectangle(
        #     frame,
        #     (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10),
        #     (int(mask[0][0]) + text_size[0] // 2 + 5, int(mask[0][1] + 5)),
        #     mask_color,
        #     -1,
        # )

        cv2.putText(
            frame, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1]) - 5), 0, 0.7, mask_color, 2
        )  
        return frame  

    # 이미지에 객체 정보 중 라벨정보(class name, confidence)를 그려주는 기능을 하는 함수      
    def draw_label(self, frame, label, x1, y1, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, font_thickness=2):

        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(frame, (int(x1), int(y1) - text_size[1]), (int(x1) + text_size[0], int(y1)), (0, 255, 0), -1)
        cv2.putText(frame, label, (int(x1), int(y1)), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        return frame
    
    def highlight_selected(self):
        selected_items = self.detected_objects_list.selectedItems()
        if selected_items:
            self.selected_track_id = selected_items[0].text()
        else:
            self.selected_track_id = None 

    def display_video(self,frame,h,w,ch):
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(frame.data, w, h,  bytesPerLine, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(convertToQtFormat)
        self.videoLabel.setPixmap(pixmap.scaled(self.videoLabel.width(), self.videoLabel.height()))

    def clear_labeled_list(self):
        self.detected_objects_list.clear()
    
    def listToString(self,str_list):
        return " ".join(str_list)
    

    # Escape, Delete 키 이벤트
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    # 메시지박스 출력 함수
    def Information_message(self, title, message) :
        info_msg = QtWidgets.QMessageBox.information(self,title,message,QtWidgets.QMessageBox.StandardButton.Yes)
        
        if info_msg == QtWidgets.QMessageBox.StandardButton.Yes:
            return
    # MainWindow maximizeButton 클릭 시 최대화 및 평균으로 크기조절 
    def maximize_restore(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = ImageViewer()
    app.exec()
    
