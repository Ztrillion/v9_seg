import seg_model_utils, cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import datetime

class DetectedObjectsWidget(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('감지된 객체 리스트')
        self.geometry('400x600')
        style = ttk.Style()
        style.configure("Treeview", font=('Helvetica', 14,'bold'))  # 글씨 크기 변경
        style.configure("Treeview.Heading", font=('Helvetica', 16, 'bold'))  # 컬럼 제목 글씨 크기 변경

        self.detected_objects = ttk.Treeview(self, columns=('Time', 'Object'), show='headings')
        self.detected_objects.heading('Time', text='Time')
        self.detected_objects.heading('Object', text='Object')
        self.detected_objects.pack(fill=tk.BOTH, expand=True)

    def add_detected_object(self, object_name):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.detected_objects.insert('', tk.END, values=(current_time, object_name))
    
    def clear_detected_objects(self):
        for i in self.detected_objects.get_children():
            self.detected_objects.delete(i)

def show_detected_video(frame,gui):
    
    bboxes_xyxy, confidences, classes, masks_xy, all_classes = general_model(frame)
    detect_draw_masks(frame, masks_xy, classes, all_classes)
    

    return frame

def detect_draw_masks(frame, masks, classes, class_names=None):
    gui.clear_detected_objects()
    person_count = sum(1 for cls in classes if class_names[int(cls)] == 'person')
    for idx, (mask, cls) in enumerate(zip(masks, classes)):
        label = f"[{idx}] {class_names[int(cls)]}"
        text_size, _ = cv2.getTextSize(label, 0, 0.7, 1)
        if int(cls) == 0 :
            pts = np.array(mask, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(frame, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1]) - 5), 0, 0.7, (0, 165, 255), 2)
        else : 
            pts = np.array(mask, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1]) - 5), 0, 0.7, (0, 255, 0), 2)
    if person_count > 0:
        object_name = f"사람이 {person_count}명 감지되었습니다."
        gui.add_detected_object(object_name)

    return frame

if __name__ == '__main__':

    gui = DetectedObjectsWidget()

    general_model = seg_model_utils.ondviceEXCUTE('lib/seg_model.onnx', conf_thres=0.5, iou_thres=0.3)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = show_detected_video(frame, gui)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        canny = cv2.Canny(blur,10,70)
        ret, mask = cv2.threshold(canny,70,255,cv2.THRESH_BINARY)
        
        gui.update_idletasks()
        gui.update()

        cv2.imshow("Video", frame)
        cv2.imshow("edge detection frame", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
