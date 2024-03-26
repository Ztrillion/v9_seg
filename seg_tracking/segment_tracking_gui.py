from PySide6 import QtCore, QtWidgets, QtGui
import cv2, os
import numpy as np
import sys
import seg_model_utils

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
        self.general_model = seg_model_utils.ondviceEXCUTE('lib/seg_model.onnx', conf_thres=0.5, iou_thres=0.3)
        self.timer = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.playVideo)
        self.selected_track_id = None
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
        self.clear_labeled_list()
        self.bboxes_xyxy, self.confidences, self.classes, self.masks_xy, self.all_classes = self.general_model(frame)
        self.bboxes_xywhn = self.xyxy_to_xywhn_array(self.bboxes_xyxy,frame.shape[1],frame.shape[0])
        self.masks_xyn = self.mask_xy_to_mask_xyn_array(self.masks_xy,frame.shape[1],frame.shape[0])
        self.object_list_results(self.classes,self.bboxes_xywhn,self.confidences)        
        self.detect_draw_masks(frame, self.masks_xy, self.classes, class_names=self.all_classes)
        #self.detect_draw_masks_tracking(frame=frame, mask=mask,track=track,mask_color=self.colors(track_id, True),track_label=f"{[track_id]}{self.general_model.names[int(cls)]}")    
        self.display_video(frame,h, w, ch)    

        return frame
    
    def object_list_results(self,classes,bboxes_xywhn,confidences):
        for idx,(cls,bbox_xywh,conf) in enumerate(zip(classes,bboxes_xywhn,confidences)):
            xn, yn, wn, hn = bbox_xywh
            item_text =f"[{idx}] {self.all_classes[int(cls)]} {xn:.3f} {yn:.3f} {wn:.3f} {hn:.3f} {conf* 100:.2f}%"
            self.detected_objects_list.addItem(item_text)       
    
    def detect_draw_masks(self, image, masks,classes,class_names=None):
        for idx,(mask, cls) in enumerate(zip(masks,classes)):
            pts = np.array(mask, np.int32).reshape((-1, 1, 2))

            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            if class_names:
                label = f"[{idx}] {class_names[int(cls)]}"
                text_size, _ = cv2.getTextSize(label, 0, 0.7, 1)
                cv2.putText(
                image, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1]) - 5), 0, 0.7, (0, 165, 255), 2)

        return image

    # def detect_draw_masks(self, frame, mask,mask_color=(255, 0, 255), track_label=None):
    #     cv2.polylines(frame, [np.int32([mask])], isClosed=True, color=mask_color, thickness=1)
    #     label = f"{track_label}" 
    #     text_size, _ = cv2.getTextSize(label, 0, 0.7, 1)

    #     cv2.putText(
    #         frame, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1]) - 5), 0, 0.7, mask_color, 2
    #     )  
    #     return frame  

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

    def xyxy_to_xywhn_array(self,bboxes_xyxy, img_w, img_h):

        bboxes_xywhn = []
        for xyxy in bboxes_xyxy:
            x1, y1, x2, y2 = xyxy
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            xc_n = round(xc / img_w, 3)
            yc_n = round(yc / img_h, 3)
            w_n = round(w / img_w, 3)
            h_n = round(h / img_h, 3)
            bboxes_xywhn.append([xc_n, yc_n, w_n, h_n])
        
        return bboxes_xywhn
    def xywhn_to_xyxy_array(self,bboxes_xywhn, img_w, img_h):

        bboxes_xywhn = []
        for xywhn in bboxes_xywhn:
            x_center, y_center, width, height = xywhn
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h
        
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            bboxes_xywhn.append([x1,y1,x2,y2])
        return bboxes_xywhn
    
    def mask_xy_to_mask_xyn_array(self,masks_xy, img_w, img_h):
        self.masks_xyn = []
        masks_xyn_append = self.masks_xyn.append
        for mask_xy in masks_xy:
            mask_xyn = [[x / img_w, y / img_h] for x, y in mask_xy]
            masks_xyn_append(mask_xyn)
        return self.masks_xyn
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
    
